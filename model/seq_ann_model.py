from . import BaseLogger
from . import Model
from . import numpy as np
from . import callbacks as cbks
from . import _make_batches,_slice_arrays
class CleanLogger(BaseLogger):
    """Callback that accumulates epoch averages of metrics.
    This callback is automatically applied to every Keras model.
    """
    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.totals = {}
        self.each_count={}
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.seen += batch_size
        for k, v in logs.items():
            if k in self.totals:
                if not np.isnan(v):
                    self.each_count[k]+=batch_size
                    self.totals[k] += v * batch_size                
            else:
                if not np.isnan(v):
                    self.each_count[k]=batch_size
                    self.totals[k] = v * batch_size
                else:
                    self.each_count[k]=0
                    self.totals[k] = 0  
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in self.params['metrics']:
                if k in self.totals:
                    # Make value available to next callbacks.
                    if self.each_count[k]!=0:   
                        logs[k] = self.totals[k] / self.each_count[k]
                    else:
                        logs[k]=np.float64('nan')
class SeqAnnModel(Model):
    def _fit_loop(self, f, ins, out_labels=None, batch_size=None,
                      epochs=100, verbose=1, callbacks=None,
                      val_f=None, val_ins=None, shuffle=True,
                      callback_metrics=None, initial_epoch=0,
                      steps_per_epoch=None, validation_steps=None):
            """Abstract fit function for `f(ins)`.
            Assume that f returns a list, labeled by out_labels.
            # Arguments
                f: Keras function returning a list of tensors
                ins: List of tensors to be fed to `f`
                out_labels: List of strings, display names of
                    the outputs of `f`
                batch_size: Integer batch size or None if unknown.
                epochs: Number of times to iterate over the data
                verbose: Verbosity mode, 0, 1 or 2
                callbacks: List of callbacks to be called during training
                val_f: Keras function to call for validation
                val_ins: List of tensors to be fed to `val_f`
                shuffle: Whether to shuffle the data at the beginning of each epoch
                callback_metrics: List of strings, the display names of the metrics
                    passed to the callbacks. They should be the
                    concatenation of list the display names of the outputs of
                     `f` and the list of display names of the outputs of `f_val`.
                initial_epoch: Epoch at which to start training
                    (useful for resuming a previous training run)
                steps_per_epoch: Total number of steps (batches of samples)
                    before declaring one epoch finished and starting the
                    next epoch. Ignored with the default value of `None`.
                validation_steps: Number of steps to run validation for
                    (only if doing validation from data tensors).
                    Ignored with the default value of `None`.
            # Returns
                `History` object.
            """
            do_validation = False
            if val_f and val_ins:
                do_validation = True
                if verbose and ins and hasattr(ins[0], 'shape') and hasattr(val_ins[0], 'shape'):
                    print('Train on %d samples, validate on %d samples' %
                          (ins[0].shape[0], val_ins[0].shape[0]))
            if validation_steps:
                do_validation = True
                if steps_per_epoch is None:
                    raise ValueError('Can only use `validation_steps` '
                                     'when doing step-wise '
                                     'training, i.e. `steps_per_epoch` '
                                     'must be set.')

            num_train_samples = self._check_num_samples(ins, batch_size,
                                                        steps_per_epoch,
                                                        'steps_per_epoch')
            if num_train_samples is not None:
                index_array = np.arange(num_train_samples)

            self.history = cbks.History()
            callbacks = [CleanLogger()] + (callbacks or []) + [self.history]
            if verbose:
                if steps_per_epoch is not None:
                    count_mode = 'steps'
                else:
                    count_mode = 'samples'
                callbacks += [cbks.ProgbarLogger(count_mode)]
            callbacks = cbks.CallbackList(callbacks)
            out_labels = out_labels or []

            # it's possible to callback a different model than self
            # (used by Sequential models)
            if hasattr(self, 'callback_model') and self.callback_model:
                callback_model = self.callback_model
            else:
                callback_model = self

            callbacks.set_model(callback_model)
            callbacks.set_params({
                'batch_size': batch_size,
                'epochs': epochs,
                'steps': steps_per_epoch,
                'samples': num_train_samples,
                'verbose': verbose,
                'do_validation': do_validation,
                'metrics': callback_metrics or [],
            })
            callbacks.on_train_begin()
            callback_model.stop_training = False
            for cbk in callbacks:
                cbk.validation_data = val_ins

            for epoch in range(initial_epoch, epochs):
                callbacks.on_epoch_begin(epoch)
                epoch_logs = {}
                if steps_per_epoch is not None:
                    for step_index in range(steps_per_epoch):
                        batch_logs = {}
                        batch_logs['batch'] = step_index
                        batch_logs['size'] = 1
                        callbacks.on_batch_begin(step_index, batch_logs)
                        outs = f(ins)

                        if not isinstance(outs, list):
                            outs = [outs]
                        for l, o in zip(out_labels, outs):
                            batch_logs[l] = o

                        callbacks.on_batch_end(step_index, batch_logs)
                        if callback_model.stop_training:
                            break

                    if do_validation:
                        val_outs = self._test_loop(val_f, val_ins,
                                                   batch_size=batch_size,
                                                   steps=validation_steps,
                                                   verbose=0)
                        if not isinstance(val_outs, list):
                            val_outs = [val_outs]
                        # Same labels assumed.
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o
                else:
                    if shuffle == 'batch':
                        index_array = _batch_shuffle(index_array, batch_size)
                    elif shuffle:
                        np.random.shuffle(index_array)

                    batches = _make_batches(num_train_samples, batch_size)
                    for batch_index, (batch_start, batch_end) in enumerate(batches):
                        batch_ids = index_array[batch_start:batch_end]
                        try:
                            if isinstance(ins[-1], float):
                                # Do not slice the training phase flag.
                                ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
                            else:
                                ins_batch = _slice_arrays(ins, batch_ids)
                        except TypeError:
                            raise TypeError('TypeError while preparing batch. '
                                            'If using HDF5 input data, '
                                            'pass shuffle="batch".')
                        batch_logs = {}
                        batch_logs['batch'] = batch_index
                        batch_logs['size'] = len(batch_ids)
                        callbacks.on_batch_begin(batch_index, batch_logs)
                        outs = f(ins_batch)
                        if not isinstance(outs, list):
                            outs = [outs]
                        for l, o in zip(out_labels, outs):
                            batch_logs[l] = o

                        callbacks.on_batch_end(batch_index, batch_logs)
                        if callback_model.stop_training:
                            break

                        if batch_index == len(batches) - 1:  # Last batch.
                            if do_validation:
                                val_outs = self._test_loop(val_f, val_ins,
                                                           batch_size=batch_size,
                                                           verbose=0)
                                if not isinstance(val_outs, list):
                                    val_outs = [val_outs]
                                # Same labels assumed.
                                for l, o in zip(out_labels, val_outs):
                                    epoch_logs['val_' + l] = o
                callbacks.on_epoch_end(epoch, epoch_logs)
                if callback_model.stop_training:
                    break
            callbacks.on_train_end()
            return self.history
