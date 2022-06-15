# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine import Config, MessageHub
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper, OptimWrapperDict
from torch import Tensor

from mmgen.core.data_structures.gen_data_sample import GenDataSample
from mmgen.registry import MODELS, MODULES
from mmgen.typing import (ForwardInputs, ForwardOutputs, LabelVar, NoiseVar,
                          TrainStepInputs, ValTestStepInputs)
from ..common import (gather_log_vars, get_valid_noise_size,
                      get_valid_num_batches, label_sample_fn, noise_sample_fn,
                      set_requires_grad)

ModelType = Union[Dict, nn.Module]
TrainInput = Union[dict, Tensor]


class BaseGAN(BaseModel, metaclass=ABCMeta):
    """Base class for GAN models.

    Args:
        generator (ModelType): The config or model of the generator.
        discriminator (Optional[ModelType]): The config or model of the
            discriminator. Defaults to None.
        data_preprocessor (Optional[Union[dict, Config]]): The pre-process
            config or :class:`~mmgen.models.GANDataPreprocessor`.
        generator_steps (int): The number of times the generator is completely
            updated before the discriminator is updated. Defaults to 1.
        discriminator_steps (int): The number of times the discriminator is
            completely updated before the generator is updated. Defaults to 1.
        ema_config (Optional[Dict]): The config for generator's exponential
            moving average setting. Defaults to None.
    """

    def __init__(self,
                 generator: ModelType,
                 discriminator: Optional[ModelType] = None,
                 data_preprocessor: Optional[Union[dict, Config]] = None,
                 generator_steps: int = 1,
                 discriminator_steps: int = 1,
                 noise_size: Optional[int] = None,
                 ema_config: Optional[Dict] = None):
        super().__init__(data_preprocessor=data_preprocessor)

        # get valid noise_size
        self.noise_size = get_valid_noise_size(noise_size, generator)

        # build generator
        if isinstance(generator, dict):
            self._gen_cfg = deepcopy(generator)
            # build generator with default `noise_size` and `num_classes`
            gen_args = dict(noise_size=self.noise_size)
            if hasattr(self, 'num_classes'):
                gen_args['num_classes'] = self.num_classes
            generator = MODULES.build(generator, default_args=gen_args)
        self.generator = generator

        # build discriminator
        if discriminator:
            if isinstance(discriminator, dict):
                self._disc_cfg = deepcopy(discriminator)
                # build discriminator with default `num_classes`
                disc_args = dict()
                if hasattr(self, 'num_classes'):
                    disc_args['num_classes'] = self.num_classes
                discriminator = MODULES.build(
                    discriminator, default_args=disc_args)
        self.discriminator = discriminator

        self._gen_steps = generator_steps
        self._disc_steps = discriminator_steps

        if ema_config is None:
            self._ema_config = None
            self._with_ema_gen = False
        else:
            self._ema_config = deepcopy(ema_config)
            self._init_ema_model(self._ema_config)
            self._with_ema_gen = True

    def noise_fn(self, noise: NoiseVar = None, num_batches: int = 1):
        """Sampling function for noise. There are three scenarios in this
        function:

        - If `noise` is a callable function, sample `num_batches` of noise
          with passed `noise`.
        - If `noise` is `None`, sample `num_batches` of noise from gaussian
          distribution.
        - If `noise` is a `torch.Tensor`, directly return `noise`.

        Args:
            noise (Union[Tensor, Callable, List[int], None]): You can directly
                give a batch of label through a ``torch.Tensor`` or offer a
                callable function to sample a batch of label data. Otherwise,
                the ``None`` indicates to use the default noise sampler.
                Defaults to `None`.
            num_batches (int, optional): The number of batches label want to
                sample. If `label` is a Tensor, this will be ignored. Defaults
                to 1.

        Returns:
            Tensor: Sampled noise tensor.
        """
        return noise_sample_fn(
            noise=noise,
            num_batches=num_batches,
            noise_size=self.noise_size,
            device=self.device)

    @property
    def generator_steps(self) -> int:
        """int: The number of times the generator is completely updated before
        the discriminator is updated."""
        return self._gen_steps

    @property
    def discriminator_steps(self) -> int:
        """int: The number of times the discriminator is completely updated
        before the generator is updated."""
        return self._disc_steps

    @property
    def device(self) -> torch.device:
        """Get current device of the model.

        Returns:
            torch.device: The current device of the model.
        """
        return next(self.parameters()).device

    @property
    def with_ema_gen(self) -> bool:
        """Whether the GAN adopts exponential moving average.

        Returns:
            bool: If `True`, means this GAN model is adopted to exponential
                moving average and vice versa.
        """
        return self._with_ema_gen

    def _init_ema_model(self, ema_config: dict):
        """Initialize a EMA model corresponding to the given `ema_config`. If
        `ema_config` is an empty dict or `None`, EMA model will not be
        initialized.

        Args:
            ema_config (dict): Config to initialize the EMA model.
        """
        ema_config.setdefault('type', 'ExponentialMovingAverage')
        self.generator_ema = MODELS.build(
            ema_config, default_args=dict(model=deepcopy(self.generator)))

    def _get_valid_mode(self,
                        batch_inputs: ForwardInputs,
                        mode: Optional[str] = None) -> str:
        """Try get the valid forward mode from inputs.

        - If forward mode is contained in `batch_inputs` and `mode` is passed
          to :meth:`forward`, we check whether that the two have the same
          value, and then return the value. If not same, `AssertionError` will
          be raised.
        - If forward mode is defined by one of `batch_inputs` or `mode`, it
          will be used as forward mode.
        - If forward mode is not defined by any input, 'ema' will returned if
          :property:`with_ema_gen` is true. Otherwise, 'orig' will be returned.

        Args:
            batch_inputs (ForwardInputs): Inputs passed to :meth:`forward`.
            mode (Optional[str]): Forward mode passed to :meth:`forward`.

        Returns:
            str: Forward mode to generate image. ('orig' or 'ema').
        """
        if isinstance(batch_inputs, dict):
            input_mode = batch_inputs.get('mode', None)
        else:  # batch_inputs is a Tensor
            input_mode = None
        if input_mode is not None and mode is not None:
            assert input_mode == mode, (
                '\'mode\' in \'batch_inputs\' is inconsistency with \'mode\'. '
                f'Receive \'{input_mode}\' and \'{mode}\'.')
        mode = input_mode or mode

        # set default value
        if mode is None:
            if self.with_ema_gen:
                mode = 'ema'
            else:
                mode = 'orig'

        # security checking for mode
        assert mode in ['ema', 'ema/orig', 'orig'
                        ], ('Only support \'ema\', \'ema/orig\', \'orig\' '
                            f'in {self.__class__.__name__}\'s image sampling.')
        if mode in ['ema', 'ema/orig']:
            assert self.with_ema_gen, (
                f'\'{self.__class__.__name__}\' do not have EMA model.')
        return mode

    def forward(self,
                batch_inputs: ForwardInputs,
                data_samples: Optional[list] = None,
                mode: Optional[str] = None) -> ForwardOutputs:
        """Sample images with the given inputs. If forward mode is 'ema' or
        'orig', the image generated by corresponding generator will be
        returned. If forward mode is 'ema/orig', images generated by original
        generator and EMA generator will both be returned in a dict.

        Args:
            batch_inputs (ForwardInputs): Dict containing the necessary
                information (e.g. noise, num_batches, mode) to generate image.
            data_samples (Optional[list]): Data samples collated by
                :attr:`data_preprocessor`. Defaults to None.
            mode (Optional[str]): Which generator used to generate images. If
                is passed, must be one of 'ema', 'orig' and 'ema/orig'.
                Defaults to None.

        Returns:
            ForwardOutputs: Generated images or image dict.
        """
        if isinstance(batch_inputs, Tensor):
            noise = batch_inputs
        else:
            if 'noise' in batch_inputs:
                noise = batch_inputs['noise']
            else:
                num_batches = get_valid_num_batches(batch_inputs)
                noise = self.noise_fn(num_batches=num_batches)

        mode = self._get_valid_mode(batch_inputs, mode)
        if mode in ['ema', 'ema/orig']:
            generator = self.generator_ema
        else:  # mode is 'orig'
            generator = self.generator

        outputs = generator(noise, return_noise=False)

        if mode == 'ema/orig':
            generator = self.generator
            outputs_orig = generator(noise, return_noise=False)
            outputs = dict(ema=outputs, orig=outputs_orig)

        return outputs

    def val_step(self, data: ValTestStepInputs) -> ForwardOutputs:
        """Gets the generated image of given data.

        Calls ``self.data_preprocessor(data)`` and
        ``self(inputs, data_sample, mode=None)`` in order. Return the
        generated results which will be passed to evaluator.

        Args:
            data (ValTestStepInputs): Data sampled from metric specific
                sampler. More detials in `Metrics` and `Evaluator`.

        Returns:
            ForwardOutputs: Generated image or image dict.
        """
        inputs_dict, data_sample = self.data_preprocessor(data)
        outputs = self(inputs_dict, data_sample)
        return outputs

    def test_step(self, data: ValTestStepInputs) -> ForwardOutputs:
        """Gets the generated image of given data. Same as :meth:`val_step`.

        Args:
            data (ValTestStepInputs): Data sampled from metric specific
                sampler. More detials in `Metrics` and `Evaluator`.

        Returns:
            ForwardOutputs: Generated image or image dict.
        """
        inputs_dict, data_sample = self.data_preprocessor(data)
        outputs = self(inputs_dict, data_sample)
        return outputs

    def train_step(self, data: TrainStepInputs,
                   optim_wrapper: OptimWrapperDict) -> Dict[str, Tensor]:
        """Train GAN model. In the training of GAN models, generator and
        discriminator are updated alternatively. In MMGeneration's design,
        `self.train_step` is called with data input. Therefore we always update
        discriminator, whose updating is relay on real data, and then determine
        if the generator needs to be updated based on the current number of
        iterations. More details about whether to update generator can be found
        in :meth:`should_gen_update`.

        Args:
            data (List[dict]): Data sampled from dataloader.
            optim_wrapper (OptimWrapperDict): OptimWrapperDict instance
                contains OptimWrapper of generator and discriminator.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')
        inputs_dict, data_sample = self.data_preprocessor(data, True)

        disc_optimizer_wrapper: OptimWrapper = optim_wrapper['discriminator']
        disc_accu_iters = disc_optimizer_wrapper._accumulative_counts

        with disc_optimizer_wrapper.optim_context(self.discriminator):
            log_vars = self.train_discriminator(inputs_dict, data_sample,
                                                disc_optimizer_wrapper)

        # add 1 to `curr_iter` because iter is updated in train loop.
        # Whether to update the generator. We update generator with
        # discriminator is fully updated for `self.n_discriminator_steps`
        # iterations. And one full updating for discriminator contains
        # `disc_accu_counts` times of grad accumulations.
        if (curr_iter + 1) % (self.discriminator_steps * disc_accu_iters) == 0:
            set_requires_grad(self.discriminator, False)
            gen_optimizer_wrapper = optim_wrapper['generator']
            gen_accu_iters = gen_optimizer_wrapper._accumulative_counts

            log_vars_gen_list = []
            # init optimizer wrapper status for generator manually
            gen_optimizer_wrapper.initialize_count_status(
                self.generator, 0, self.generator_steps * gen_accu_iters)
            for _ in range(self.generator_steps * gen_accu_iters):
                with gen_optimizer_wrapper.optim_context(self.generator):
                    log_vars_gen = self.train_generator(
                        inputs_dict, data_sample, gen_optimizer_wrapper)

                log_vars_gen_list.append(log_vars_gen)
            log_vars_gen = gather_log_vars(log_vars_gen_list)
            log_vars_gen.pop('loss', None)  # remove 'loss' from gen logs

            set_requires_grad(self.discriminator, True)

            # only do ema after generator update
            if self.with_ema_gen:
                self.generator_ema.update_parameters(self.generator)

            log_vars.update(log_vars_gen)

        return log_vars

    @abstractmethod
    def train_generator(self, inputs: TrainInput,
                        data_samples: List[GenDataSample],
                        optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        """Training function for discriminator. All GANs should implement this
        function by themselves.

        Args:
            inputs (TrainInput): Inputs from dataloader.
            data_samples (List[GenDataSample]): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """

    @abstractmethod
    def train_discriminator(
            self, inputs: TrainInput, data_samples: List[GenDataSample],
            optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        """Training function for discriminator. All GANs should implement this
        function by themselves.

        Args:
            inputs (TrainInput): Inputs from dataloader.
            data_samples (List[GenDataSample]): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """


class BaseConditionalGAN(BaseGAN):
    """Base class for Conditional GAM models.

    Args:
        generator (ModelType): The config or model of the generator.
        discriminator (Optional[ModelType]): The config or model of the
            discriminator. Defaults to None.
        data_preprocessor (Optional[Union[dict, Config]]): The pre-process
            config or :class:`~mmgen.models.GANDataPreprocessor`.
        generator_steps (int): The number of times the generator is completely
            updated before the discriminator is updated. Defaults to 1.
        discriminator_steps (int): The number of times the discriminator is
            completely updated before the generator is updated. Defaults to 1.
        noise_size (Optional[int]): Size of the input noise vector.
            Default to None.
        num_classes (Optional[int]): The number classes you would like to
            generate. Defaults to None.
        ema_config (Optional[Dict]): The config for generator's exponential
            moving average setting. Defaults to None.
    """

    def __init__(self,
                 generator: ModelType,
                 discriminator: Optional[ModelType] = None,
                 data_preprocessor: Optional[Union[dict, Config]] = None,
                 generator_steps: int = 1,
                 discriminator_steps: int = 1,
                 noise_size: Optional[int] = None,
                 num_classes: Optional[int] = None,
                 ema_config: Optional[Dict] = None):

        self.num_classes = self._get_valid_num_classes(num_classes, generator,
                                                       discriminator)
        super().__init__(generator, discriminator, data_preprocessor,
                         generator_steps, discriminator_steps, noise_size,
                         ema_config)

    def label_fn(self, label: LabelVar = None, num_batches: int = 1) -> Tensor:
        """Sampling function for label. There are three scenarios in this
        function:

        - If `label` is a callable function, sample `num_batches` of labels
          with passed `label`.
        - If `label` is `None`, sample `num_batches` of labels in range of
          `[0, self.num_classes-1]` uniformly.
        - If `label` is a `torch.Tensor`, check the range of the tensor is in
          `[0, self.num_classes-1]`. If all values are in valid range,
          directly return `label`.

        Args:
            label (Union[Tensor, Callable, List[int], None]): You can directly
                give a batch of label through a ``torch.Tensor`` or offer a
                callable function to sample a batch of label data. Otherwise,
                the ``None`` indicates to use the default label sampler.
                Defaults to `None`.
            num_batches (int, optional): The number of batches label want to
                sample. If `label` is a Tensor, this will be ignored. Defaults
                to 1.

        Returns:
            Tensor: Sampled label tensor.
        """
        return label_sample_fn(
            label=label,
            num_batches=num_batches,
            num_classes=self.num_classes,
            device=self.device)

    def data_sample_to_label(self,
                             data_sample: list) -> Optional[torch.Tensor]:
        """Get labels from input `data_sample` and pack to `torch.Tensor`. If
        no label is found in the passed `data_sample`, `None` would be
        returned.

        Args:
            data_sample (List[InstanceData]): Input data samples.

        Returns:
            Optional[torch.Tensor]: Packed label tensor.
        """
        # assume all data_sample have the same data fields
        if not data_sample or 'gt_label' not in data_sample[0].keys():
            return None
        gt_labels = [sample.gt_label.label for sample in data_sample]
        gt_labels = torch.cat(gt_labels, dim=0)
        return gt_labels

    @staticmethod
    def _get_valid_num_classes(num_classes: Optional[int],
                               generator: ModelType,
                               discriminator: Optional[ModelType]) -> int:
        """Try to get the value of `num_classes` from input, `generator` and
        `discriminator` and check the consistency of these values. If no
        conflict is found, return the `num_classes`.

        Args:
            num_classes (Optional[int]): `num_classes` passed to
                `BaseConditionalGAN_refactor`'s initialize function.
            generator (ModelType): The config or the model of generator.
            discriminator (Optional[ModelType]): The config or model of
                discriminator.

        Returns:
            int: The number of classes to be generated.
        """
        if isinstance(generator, dict):
            num_classes_gen = generator.get('num_classes', None)
        else:
            num_classes_gen = getattr(generator, 'num_classes', None)

        num_classes_disc = None
        if discriminator is not None:
            if isinstance(discriminator, dict):
                num_classes_disc = discriminator.get('num_classes', None)
            else:
                num_classes_disc = getattr(discriminator, 'num_classes', None)

        # check consistency between gen and disc
        if num_classes_gen is not None and num_classes_disc is not None:
            assert num_classes_disc == num_classes_gen, (
                '\'num_classes\' is unconsistency between generator and '
                f'discriminator. Receive \'{num_classes_gen}\' and '
                f'\'{num_classes_disc}\'.')
        model_num_classes = num_classes_gen or num_classes_disc

        if num_classes is not None and model_num_classes is not None:
            assert num_classes == model_num_classes, (
                'Input \'num_classes\' is unconsistency with '
                f'model\'s ones. Receive \'{num_classes}\' and '
                f'\'{model_num_classes}\'.')

        num_classes = num_classes or model_num_classes
        return num_classes

    def forward(self,
                batch_inputs: ForwardInputs,
                data_samples: Optional[list] = None,
                mode: Optional[str] = None) -> ForwardOutputs:
        """Sample images with the given inputs. If forward mode is 'ema' or
        'orig', the image generated by corresponding generator will be
        returned. If forward mode is 'ema/orig', images generated by original
        generator and EMA generator will both be returned in a dict.

        Args:
            batch_inputs (ForwardInputs): Dict containing the necessary
                information (e.g. noise, num_batches, mode) to generate image.
            data_samples (Optional[list]): Data samples collated by
                :attr:`data_preprocessor`. Defaults to None.
            mode (Optional[str]): Which generator used to generate images. If
                is passed, must be one of 'ema', 'orig' and 'ema/orig'.
                Defaults to None.

        Returns:
            ForwardOutputs: Generated images or image dict.
        """
        if isinstance(batch_inputs, Tensor):
            noise = batch_inputs
        else:
            if 'noise' in batch_inputs:
                noise = batch_inputs['noise']
            else:
                num_batches = get_valid_num_batches(batch_inputs)
                noise = self.noise_fn(num_batches=num_batches)

        labels = self.data_sample_to_label(data_samples)
        if labels is None:
            num_batches = get_valid_num_batches(batch_inputs)
            labels = self.label_fn(num_batches=num_batches)

        mode = self._get_valid_mode(batch_inputs, mode)
        if mode in ['ema', 'ema/orig']:
            generator = self.generator_ema
        elif mode == 'orig':
            generator = self.generator
        outputs = generator(noise, label=labels, return_noise=False)

        if mode == 'ema/orig':
            generator = self.generator
            outputs_orig = generator(noise, label=labels, return_noise=False)

            outputs = dict(ema=outputs, orig=outputs_orig)

        return outputs
