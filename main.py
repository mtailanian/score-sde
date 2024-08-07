# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Training and evaluation"""

"""
NEW: 
python main.py --mode train --dataset mvtec --category carpet
python main.py --mode train --dataset visa --category candle
------------------------------------------------------------------------------------------------------------------------
OLD: 
Magic: python main.py --mode train --config configs/ve/anomaly_256_ncsnpp_continuous.py --workdir /data/tai/phd/training/sde/hazelnut --eval_folder evaluation
Hercules: python main.py --mode train --config configs/ve/anomaly_256_ncsnpp_continuous.py --workdir /home/data/tai/phd/training/sde/hazelnut --eval_folder evaluation 
"""

import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import tensorflow as tf
from path_utils import get_path
from pathlib import Path


# SDE = 've'
SDE = 'vp'
# SDE = 'subvp'

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", f"configs/{SDE}/anomaly_256_ncsnpp_continuous.py", "Training configuration.", lock_config=False)

flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("dataset", None, "Dataset.")
flags.DEFINE_string("category", None, "Category from dataset.")
flags.DEFINE_string("workdir", str(get_path('training') / 'dad' / SDE), "Work directory.")
flags.DEFINE_string("eval_folder", "evaluation", "The folder name for storing evaluation results")

flags.mark_flags_as_required(["dataset", "category", "workdir", "config", "mode"])

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def main(argv):

  FLAGS.config.data.dataset = FLAGS.dataset
  FLAGS.config.data.category = FLAGS.category

  if FLAGS.mode == "train":
    # Create the working directory
    tf.io.gfile.makedirs(FLAGS.workdir)
    # Set logger so that it outputs to both console and file
    # Make logging work for both disk and Google Cloud Storage
    gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    # Run the training pipeline
    run_lib.train(FLAGS.config, str(Path(FLAGS.workdir) / FLAGS.dataset / FLAGS.category))
  elif FLAGS.mode == "eval":
    # Run the evaluation pipeline
    run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":

  app.run(main)
