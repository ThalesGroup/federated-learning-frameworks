# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import tensorflow as tf
from tf2_net import Net
import numpy as np
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, make_model_learnable,ModelLearnableKey,model_learnable_to_dxo
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants
from nvflare.fuel.utils import fobs
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
scaler=MinMaxScaler()


class TF2ModelPersistor(ModelPersistor):
    def __init__(self, save_name="tf2_model.fobs"):
        super().__init__()
        self.save_name = save_name
        self.test_images, self.test_labels = None, None

    def _initialize(self, fl_ctx: FLContext):
        train = pd.read_csv("/workspace/poc/admin/transfer/IDD_MLP_Tensorflow/jobs/IDD_MLP_Tensorflow/app/custom/my_data_result/train_data.csv")
        X_train_scaled = scaler.fit_transform(train.drop(['y'], axis=1).to_numpy())
        y_train = train['y'].to_numpy()
        test=pd.read_csv("/workspace/poc/admin/transfer/IDD_MLP_Tensorflow/jobs/IDD_MLP_Tensorflow/app/custom/my_data_result/test_data.csv")
        X_test_scaled = scaler.transform(test.drop(['y'], axis=1).to_numpy())
        y_test = test['y'].to_numpy()
        y_test_cat = to_categorical(y_test)


        self.test_images = X_test_scaled
        self.test_labels=y_test_cat
        # get save path from FLContext
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        env = None
        run_args = fl_ctx.get_prop(FLContextKey.ARGS)
        if run_args:
            env_config_file_name = os.path.join(app_root, run_args.env)
            if os.path.exists(env_config_file_name):
                try:
                    with open(env_config_file_name) as file:
                        env = json.load(file)
                except:
                    self.system_panic(
                        reason="error opening env config file {}".format(env_config_file_name), fl_ctx=fl_ctx
                    )
                    return

        if env is not None:
            if env.get("APP_CKPT_DIR", None):
                fl_ctx.set_prop(AppConstants.LOG_DIR, env["APP_CKPT_DIR"], private=True, sticky=True)
            if env.get("APP_CKPT") is not None:
                fl_ctx.set_prop(
                    AppConstants.CKPT_PRELOAD_PATH,
                    env["APP_CKPT"],
                    private=True,
                    sticky=True,
                )

        log_dir = fl_ctx.get_prop(AppConstants.LOG_DIR)
        if log_dir:
            self.log_dir = os.path.join(app_root, log_dir)
        else:
            self.log_dir = app_root
        self._fobs_save_path = os.path.join(self.log_dir, self.save_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        fl_ctx.sync_sticky()

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        """Initializes and loads the Model.

        Args:
            fl_ctx: FLContext

        Returns:
            Model object
        """

        if os.path.exists(self._fobs_save_path):
            self.logger.info("Loading server weights")
            with open(self._fobs_save_path, "rb") as f:
                model_learnable = fobs.load(f)
        else:
            self.logger.info("Initializing server model")
            self.network = Net()
            #loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self.network.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])
            _ = self.network(tf.keras.Input(shape=(77, )))
            var_dict = {self.network.get_layer(index=key).name: value for key, value in enumerate(self.network.get_weights())}
            self.var_list = [self.network.get_layer(index=index).name for index in range(len(self.network.get_weights()))]
            model_learnable = make_model_learnable(var_dict, dict())
        return model_learnable

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)

    def save_model(self, model_learnable: ModelLearnable, fl_ctx: FLContext):
        """Saves model.

        Args:
            model_learnable: ModelLearnable object
            fl_ctx: FLContext
        """


        dxo=model_learnable_to_dxo(model_learnable)
        model_weights = dxo.data
        prev_weights = {
            self.network.get_layer(index=key).name: value for key, value in enumerate(self.network.get_weights())
        }

        ordered_model_weights = {key: model_weights.get(key) for key in prev_weights}
        for key in self.var_list:
            value = ordered_model_weights.get(key)
            if np.all(value == 0):
                ordered_model_weights[key] = prev_weights[key]

        # update local model weights with received weights
        self.network.set_weights(list(ordered_model_weights.values()))
        
        loss, accuracy =self.network.evaluate(self.test_images, self.test_labels, verbose=0)
        print("==**==**== Test loss:",loss,"  ==**==**==  Test accuracy:", accuracy)
        
        model_learnable_info = {k: str(type(v)) for k, v in model_learnable.items()}
        self.logger.info(f"Saving aggregated server weights: \n {model_learnable_info}")
        with open(self._fobs_save_path, "wb") as f:
            fobs.dump(model_learnable, f)


