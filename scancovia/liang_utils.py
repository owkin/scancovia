"""
Code to load the models for the paper of Lian et al.
Source: https://github.com/cojocchen/covid19_critically_ill/blob/master/prediction.py
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path

FEATURE_LIST = ['Number.of.comorbidities',
                'Lactate.dehydrogenase',
                'Age',
                'NLR',
                'Creatine.kinase',
                'Direct.bilirubin',
                'Malignancy',
                'X.ray.abnormality',
                'COPD',
                'Dyspnea', ]

MODEL_PATH = Path(__file__).parents[1] / 'assets/models_liang/'


class Model(object):
    def __init__(self,
                 model_path,
                 config={}):
        self.init_model(model_path, config)
        self.set_names(config)
        # print([n.name for n in tf.get_default_graph().as_graph_def().node])

    def init_model(self, model_path, config={}):
        model_name = config.get('model_name', 'fold_0_fea_10_layer_3')
        graph_path = os.path.join(model_path, model_name + '.meta')
        saver = tf.train.import_meta_graph(graph_path)
        sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, os.path.join(model_path, model_name))

        self.sess = sess
        return

    def set_names(self, config={}):
        self.input_vars = config.get('input_vars',
                                     ['Malignancy',
                                      'X.ray.abnormality',
                                      'COPD',
                                      'Dyspnea',
                                      'Number.of.comorbidities',
                                      'Lactate.dehydrogenase',
                                      'Age',
                                      'NLR',
                                      'Creatine.kinase',
                                      'Direct.bilirubin']
                                     )
        if config.get('scope', None) is not None:
            scope_name = '%s/' % config['scope']
        else:
            scope_name = ''
        self.input_tensor_name = config.get('input_tensor_name', '%sX-Input:0' % scope_name)
        self.output_tensor_name = config.get('output_tensor_name', '%shidden_layers/layer3/Tanh:0' % scope_name)
        graph = tf.get_default_graph()
        self._input = graph.get_tensor_by_name(self.input_tensor_name)
        self._output = graph.get_tensor_by_name(self.output_tensor_name)
        self._keepprob = graph.get_tensor_by_name('%sPlaceholder:0' % scope_name)

    def predict(self, val_dict):
        X_list = []
        for vname in self.input_vars:
            assert vname in val_dict
            X_list.append(val_dict[vname])
        X_np = np.array(X_list)
        if len(X_np.shape) == 1:
            X_np = X_np.reshape((1, -1))
        else:
            X_np = X_np.transpose()
        risk_score = self.sess.run(self._output, feed_dict={self._input: X_np, self._keepprob: 1.0})
        return risk_score


class Model_NFold(object):
    def __init__(self, model_path,
                 config={}):
        """
        Arguments:
            model_path is the path containing the fold_0, fold_1, etc folders
            config: can leave it blank to use the default values
        """
        self.config = config
        self.fill_default_config()
        self.init_models(model_path)

    def fill_default_config(self):
        self.config['model_name'] = self.config.get('model_name', 'model')
        self.config['fold_list'] = self.config.get('fold_list', ['fold_%d' % x for x in range(5)])

    def init_models(self, model_path):
        tf.reset_default_graph()
        self.models = {}
        for fold in self.config['fold_list']:
            basename_model = os.path.join(model_path,
                                          '%s' % fold)
            scope_name = os.path.basename(model_path) + '_%s' % fold
            scope_name = f'fold_{fold}'
            with tf.variable_scope(scope_name):
                self.config['scope'] = scope_name
                self.models[scope_name] = Model(basename_model, self.config)

    def predict(self, val_dict):
        res = None
        for fold in self.models:
            with tf.variable_scope(fold):
                result = self.models[fold].predict(val_dict)
            if res is None:
                res = result
            else:
                res += result
        res /= len(self.config['fold_list'])
        return res


class Model_COX_DL(object):
    def __init__(self, config={}):
        """
        Arguments:
            model_path is the path containing the fold_0, fold_1, etc folders
            config: can leave it blank to use the default values
        """
        self.set_params(config)

    def set_params(self, config={}):
        self.input_vars = config.get('input_vars_and_coef',
                                     [('Malignancy', 1.06741622),
                                      ('X.ray.abnormality', 0.65624252),
                                      ('COPD', 0.28010291),
                                      ('Dyspnea', 0.34496069),
                                      ('Number.of.comorbidities', 0.06626169),
                                      ('Lactate.dehydrogenase', 0.04888803),
                                      ('Age', 0.09156370),
                                      ('NLR', 0.02040950),
                                      ('Creatine.kinase', 0.05368482),
                                      ('Direct.bilirubin', 0.03486548),
                                      ('DL.feature', 1.50393931)]
                                     )
        self.cumulative_base_hazard = config.get('cumulative_base_hazard',
                                                 [('5days', 0.03552347),
                                                  ('10days', 0.04164459),
                                                  ('30days', 0.04840961),
                                                  ]
                                                 )

    def predict(self, val_dict):
        lpnew = None
        for var, coef in self.input_vars:
            if lpnew is None:
                lpnew = val_dict[var] * coef
            else:
                lpnew += val_dict[var] * coef
        prob = {}
        for day, coef in self.cumulative_base_hazard:
            prob[day] = 1 - np.exp(-np.exp(lpnew) * coef)
        return {'prob': prob, 'score': lpnew}
