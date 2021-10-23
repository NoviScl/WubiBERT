'''
A class encapsulating configuration for job submission
'''
import os
import consts


class Job:
    '''
    A class encapsulating configuration of a job, and methods for geenerating
    environment for job submission (but doesn't do submission itself).
    '''
    def __init__(self, config: dict):
        '''
        Example config:
        ```
        {
            'task': 'tnews',
            'tokenizer': 'pinyin',
            'ckpt': 'ckpt_8840',
            'seed': 10,
            'debug': False,
            'two_level_embeddings': False,
            'use_shuffled': False,
            'use_no_index': False,
            'use_cws': False,
            'use_500': False,
            'max_seq_len': 128,
            'do_train': True,
            'do_test': True,
        }
        ```
        '''
        self.config = config
        
        # Generate other variables
        self.tokenizer_type = self.get_tokenizer_type()
        self.vocab_file = self.get_vocab_file()
        self.vocab_model_file = self.vocab_file.replace('.vocab', '.model')
        if self.config['use_cws']:
            self.cws_vocab_file = self.vocab_file.replace('.vocab', '.cws_vocab')
        self.train_dir, self.dev_dir, self.test_dir = self.get_data_dirs()
        self.data_dir = self.test_dir
        self.config_file = self.get_config_file()
        self.dir_ckpts = self.get_dir_ckpts()
        self.init_checkpoint = os.path.join(self.dir_ckpts, 
                                            self.config['ckpt'] + '.pt')
        self.output_dir = self.get_output_dir()
        if 'noise_type' in self.config:
            self.test_model = self.get_test_model()
        self.mode = self.get_mode()
        self.script = self.get_script()

    def get_tokenizer_type(self):
        if self.config['tokenizer'] == 'pinyin_concat_wubi':
            return 'PinyinConcatWubi'
        elif self.config['use_no_index']:
            return 'CommonZhNoIndex'
        elif self.config['use_shuffled']:
            return 'Shuffled'
        elif self.config['use_cws']:
            return 'CWS'
        else:
            return consts.TOKENIZER_TYPES[self.config['tokenizer']]

    def get_vocab_file(self):
        if self.config['use_no_index']:
            return consts.VOCAB_FILES_NO_INDEX[self.config['tokenizer']]
        if self.config['use_shuffled']:
            if self.config['use_500']:
                return consts.VOCAB_FILES_SHUFFLED_500[self.config['tokenizer']]
            return consts.VOCAB_FILES_SHUFFLED[self.config['tokenizer']]
        if self.config['use_500']:
            return consts.VOCAB_FILES_500[self.config['tokenizer']]
        if self.config['use_cws']:
            return consts.VOCAB_FILES_CWS[self.config['tokenizer']].format('80')
        return consts.VOCAB_FILES[self.config['tokenizer']]

    def is_classification_task(self) -> bool:
        C_TASKS = [
            'tnews'
            'iflytek',
            'wsc',
            'afqmc',
            'csl',
            'ocnli',
            'bq',
            'lcqmc',
            'thucnews',
        ]
        return self.config['task'] in C_TASKS

    def get_data_dirs(self):
        '''
        Return train_dir, dev_dir, test_dir
        '''
        def paths_append(paths, suf):
            for i in range(len(paths)):
                paths[i] = os.path.join(paths[i], suf)
            return paths

        dirs = [''] * 3
        dirs = paths_append(dirs, 'datasets')
        dirs = paths_append(dirs, self.config['task'])

        if 'noise_type' in self.config:
            # Use noisy data
            if self.config['noise_train'] > 0:
                raise NotImplementedError
            if self.config['noise_test'] > 0:
                dirs[2] = os.path.join(
                    dirs[2], 
                    'noisy', 
                    self.config['noise_type'] + '_' + str(self.config['noise_test']))
            else:
                if self.fewshot:
                    raise NotImplementedError
                else:
                    # Some tasks doesn't have split directories
                    if self.config['task'] not in ['lcqmd', 'bq', 'thucnews']:
                        dirs[2] = os.path.join(dirs[2], 'split')
        else:
            # Not noisy, handle split dir
            # Some tasks (LCQMD, BQ, THUCNEWS etc) don't have split directories
            if self.config['task'] not in ['lcqmc', 'bq', 'thucnews']:
                dirs = paths_append(dirs, 'split')
        return tuple(dirs)

    def get_config_file(self):
        if 'use_base' in self.config and self.config['use_base']:
            return os.path.join('configs', 'bert_base_config.json')
        elif self.config['use_500']:
            return os.path.join('configs', 'bert_config_vocab500.json')
        else:
            return os.path.join('configs', 'bert_config_vocab22675.json')

    def get_dir_ckpts(self):
        if 'use_sp' in self.config and self.config['use_sp']:
            return consts.DIR_CKPT_SP[self.config['tokenizer']]
        elif 'use_long' in self.config and self.config['use_long']:
            return consts.DIR_CKPTS_LONG[self.config['tokenizer']]
        elif self.config['use_no_index']:
            return consts.DIR_CKPTS_NO_INDEX[self.config['tokenizer']]
        elif self.config['use_shuffled']:
            if self.config['use_500']:
                return consts.DIR_CKPTS_SHUFFLED_500[self.config['tokenizer']]
            else:
                return consts.DIR_CKPTS_SHUFFLED[self.config['tokenizer']]
        elif self.config['use_500']:
            return consts.DIR_CKPTS_500[self.config['tokenizer']]
        elif self.config['use_cws']:
            return consts.DIR_CKPTS_CWS[self.config['tokenizer']]
        else:
            return consts.DIR_CKPTS[self.config['tokenizer']]

    def get_ckpt(self):
        if self.config['use_sp']:
            return 'ckpt_8601'
        elif self.config['use_base']:
            raise NotImplementedError
        elif self.config['use_500']:
            return consts.BSET_CKPTS_500[self.config['tokenizer']]
        elif self.config['use_no_index']:
            return consts.BEST_CKPTS_NO_INDEX[self.config['tokenizer']]
        elif self.config['tokenizer'] == 'pinyin_concat_wubi':
            raise NotImplementedError
        elif self.config['use_cws']:
            return consts.BEST_CKPTS_CWS[self.config['tokenizer']]
        else:
            return consts.BEST_CKPTS[self.config['tokenizer']]

    def get_epochs(self):
        if self.config['task'] == 'wsc':
            return 24
        if self.config['task'] == 'thucnews':
            return 4
        if self.config['task'] == 'cluener':
            return 12
        else:
            return 6

    def get_output_dir(self):
        task = self.config['task']
        tokenizer = self.config['tokenizer']
        if 'fewshot' in self.config and self.cofnig['fewshot']:
            task += '_fewshot'
        
        if 'use_sp' in self.config and self.config['use_sp']:
            output_dir = os.path.join('logs', task, 'sp', tokenizer)
        else:
            if 'noise_type' in self.config:
                task += '_{}_{}_{}'.format(self.config['noise_type'],
                                           self.config['noise_train'],
                                           self.config['noise_test'])
            
            if 'use_base' in self.config and self.config['use_base']:
                tokenizer += '_base'
            if 'use_long' in self.config and self.config['use_long']:
                tokenizer += '_long'
            if self.config['use_shuffled']:
                tokenizer += '_shuffled'
            if self.config['use_no_index']:
                tokenizer += '_no_index'
            if self.config['use_cws']:
                tokenizer += '_cws'
            if self.config['use_500']:
                tokenizer += '_500'
            if self.config['two_level_embeddings']:
                tokenizer += '_twolevel'
            if 'max_seq_len' in self.config:
                tokenizer += '_seqlen' + str(self.config['max_seq_len'])

            if task == 'drcd':
                tokenizer += '_trad'  # DRCD always use traditional Chinese

            output_dir = os.path.join('logs', task, tokenizer)
        output_dir = os.path.join(output_dir, self.config['ckpt'])
        return output_dir

    def get_test_model(self):
        noise_task = '_{}_{}_{}'.format(self.config['noise_type'],
                                        self.config['noise_train'],
                                        self.config['noise_test'])
        clean_dir = self.output_dir.replace(noise_task, '')
        best_model = os.path.join(clean_dir, str(self.config['seed']), 'best_model.bin')
        return best_model

    def get_mode(self):
        mode = []
        if self.config['do_train']:
            mode += ['train', 'eval']
        if self.config['do_test']:
            mode += ['test']
        return ' '.join(mode)

    def get_script(self):
        if self.config['task'] in ['chid', 'c3', 'cmrc']:
            filename = 'run_mrc_' + self.config['task'] + '.sh'
            return os.path.join('scripts', filename)
        elif self.config['task'] == 'drcd':
            return os.path.join('scripts', 'run_mrc_cmrc.sh')
        elif self.config['task'] == 'cluener':
            return os.path.join('scripts', 'run_ner.sh')
        else:
            return os.path.join('scripts', 'run_finetune.sh')
        
    def set_task_specific_settings(self):
        raise NotImplementedError

    def print_vars(self):
        print(f'self.script = {self.script}')
        print(f'self.config[\'task\'] = {self.config["task"]}')
        print(f'self.config[\'tokenizer\'] = {self.config["tokenizer"]}')
        print(f'self.init_checkpoint = {self.init_checkpoint}')
        print(f'self.config["seed"] = {self.config["seed"]}')
        print(f'self.vocab_file = {self.vocab_file}')
        print(f'self.train_dir = {self.train_dir}')
        print(f'self.dev_dir = {self.dev_dir}')
        print(f'self.test_dir = {self.test_dir}')
        print(f'self.output_dir = {self.output_dir}')
        if self.config['noise_type'] is not None:
            print(f'self.test_model = {self.test_model}')

    def get_vars(self):
        '''
        Return bash environment for job submission
        '''
        ret = {
            # Hyperparameters
            'max_seq_len': self.config['max_seq_len'],
            'batch_size': self.config['batch_size'],
            'epochs': self.get_epochs(),
            'seed': self.config['seed'],
            'config_file': self.config_file,   # Hyperparameters of pre-training model
            
            # Other settings
            'out_dir': self.output_dir,
            'init_checkpoint': self.init_checkpoint,
            'task_name': self.config['task'],
            'vocab_file': self.vocab_file,
            'vocab_model_file': self.vocab_model_file,
            'tokenizer_type': self.tokenizer_type,
            'data_dir': self.data_dir,
            'train_dir': self.train_dir,
            'dev_dir': self.dev_dir,
            'test_dir': self.test_dir,
            # 'fewshot': str(int(self.config['fewshot'])),
            # 'convert_to_simplified': self.drcd_convert_to_simplified,
            # 'batch_size': self.batch_size,
            'mode': self.mode,
            'two_level_embeddings': str(int(self.config['two_level_embeddings'])),
            'debug': str(int(self.config['debug'])),
        }
        
        if 'noise_type' in self.config:
            ret['test_model'] = self.test_model
        if 'use_cws' in self.config and self.config['use_cws']:
            ret['cws_vocab_file'] = self.cws_vocab_file
        return ret

