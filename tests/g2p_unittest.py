
import unittest
import tensorflow as tf
import shutil
from g2p_seq2seq.g2p import G2PModel
from g2p_seq2seq.params import Params
import inspect, os

from IPython.core.debugger import Tracer

class TestG2P(unittest.TestCase):

  def test_train(self):
    model_dir = os.path.abspath("tests/models/train")
    train_path = os.path.abspath("tests/data/toydict.train")
    dev_path = os.path.abspath("tests/data/toydict.test")
    params = Params(model_dir, train_path)
    g2p_model = G2PModel(params)
    g2p_model.prepare_data(train_path=train_path, dev_path=dev_path)
    g2p_model.train()
    shutil.rmtree(model_dir)

  def test_decode(self):
    model_dir = os.path.abspath("tests/models/decode")
    decode_from_file = os.path.abspath("tests/data/toydict.graphemes")
    decode_to_file = os.path.abspath("tests/models/decode/decode_output.txt")
    params = Params(model_dir, decode_from_file)
    g2p_model = G2PModel(params)
    g2p_model.prepare_data(test_path=decode_from_file)
    g2p_model.decode(decode_from_file=decode_from_file,
      decode_to_file=decode_to_file)


  #def test_evaluate(self):
  #  model_dir = "tests/models/decode"
  #  with g2p.tf.Graph().as_default():
  #    g2p_model = g2p.G2PModel(model_dir)
  #    g2p_model.load_decode_model()
  #    test_lines = open("tests/data/toydict.test").readlines()
  #    g2p_model.evaluate(test_lines)
  #    test_dic = data_utils.collect_pronunciations(test_lines)
  #    errors = g2p_model.calc_error(test_dic)
  #    self.assertAlmostEqual(float(errors)/len(test_dic), 0.667, places=3)

  #def test_decode(self):
  #  model_dir = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), "models/decode")
  #  g2p_model = g2p.G2PModel(model_dir)
  #  g2p_params = params.Params(model_dir, decode_flag=True)
  #  #phoneme_lines = g2p_model.load_decode_model(g2p_params)
  #  g2p_model.load_decode_model(g2p_params)
  #  #decode_lines = open("tests/data/toydict.graphemes").readlines()
  #  Tracer()()
  #  phoneme_lines = g2p_model.decode(return_output_list=True)
  #  self.assertEqual(phoneme_lines[0], u'SEQUENCE_END')
  #  self.assertEqual(phoneme_lines[1], u'SEQUENCE_END')
  #  self.assertEqual(phoneme_lines[2], u'SEQUENCE_END')
