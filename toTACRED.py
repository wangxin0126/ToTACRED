import os
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from collections import Counter
import StanfordDependencies
from nltk.parse import stanford
import json

from tqdm import tqdm_notebook, tqdm

MORE_TAHN_TWO = './log/more_than_two.txt'
ERROR = './log/error.txt'

more_than_two_list = ''
error_list = ''

#for pos
#replace the path according to your situation
java_path = "/usr/bin/java"
os.environ['JAVAHOME'] = java_path
path_to_model = "../StanfordNLP/stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger"
path_to_jar = "/../StanfordNLP/stanford-postagger-2018-10-16/stanford-postagger.jar"
tagger=StanfordPOSTagger(path_to_model, path_to_jar)
tagger.java_options='-mx4096m'          ### Setting higher memory limit for long sentences

#for ner
st = StanfordNERTagger('../StanfordNLP/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
             '../StanfordNLP/stanford-ner-2018-10-16/stanford-ner.jar', encoding='utf-8')

#for deprel
os.environ['STANFORD_PARSER'] = '../StanfordNLP/stanford-parser-full-2018-10-17/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '../StanfordNLP/stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar'
parser = stanford.StanfordParser(model_path="../StanfordNLP/stanford-parser-full-2018-10-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
sd = StanfordDependencies.get_instance(jar_filename='../StanfordNLP/stanford-parser-full-2018-10-17/stanford-parser.jar')

#for read_data
train_path = './SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
test_path = './SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
#for write_json
train_json = './ToTacredResult/train.json'
test_json = './ToTacredResult/test.json'

def most_common(words):
    user_counter = Counter(words)
    if len(user_counter.most_common(len(words))) == 1:
        return user_counter.most_common(1), False
    else:
        return user_counter.most_common(1), True


def get_pos(tokens):
    res = [item[1] for item in tagger.tag(tokens)]
    return res

def get_ner(sentence):
    tokenized_text = word_tokenize(sentence)
    classified_text = st.tag(tokenized_text)
    res = [item[1] for item in classified_text]
    return res

def get_deprel(sentence):
    sentences = parser.parse((sentence,))
    for sentence in sentences:
        break
    penn_treebank_tree = str(sentence)
    converted_tree = sd.convert_tree(penn_treebank_tree)
    res_deprel = [node.deprel for node in converted_tree]
    res_head = [node.head for node in converted_tree]
    
    return res_deprel, res_head

def read_data(path):
    with open(path, 'r') as f:
        data = f.readlines()
    return data

def get_sentence(line):
    global error_list
    sentence = line.strip().split('\t')
    #print(sentence)
    subj_start = subj_end = obj_start = obj_end = 0
    #print(len(sentence))
    if len(sentence) == 2:
        return sentence[1][1:-1]
    else:
        error_list += '*'*100
        error_list += 'can not split line:{0}'.format(line) + '\n'
        
        #print('can not split line:{0}'.format(line))
        return None
    
def get_tokens(sentence):
    res = sentence.split(' ')
    subj_start = subj_end = obj_start = obj_end = 0
    tokens = []
    
    for i, word in enumerate(res):
        if '<e1>' in word:
            subj_start = i + 1
            word = word.replace('<e1>', '')
        if '</e1>' in word:
            subj_end = i + 1
            word = word.replace('</e1>', '')
        if '<e2>' in word:
            obj_start = i + 1
            word = word.replace('<e2>', '')
        if '</e2>' in word:
            obj_end = i + 1
            word = word.replace('</e2>', '')
        tokens.append(word)
            
    return tokens, subj_start, subj_end, obj_start, obj_end
    
    
def to_TACRED(path, save_path):
    global error_list, more_than_two_list
    data = read_data(path)
    
    length = len(data)
    json_datas = []
    for i in tqdm(range(0, length, 4)):
        #print(data[i])
        try:
            sentence = get_sentence(data[i])
            if sentence is None:
                continue
            relation = data[i+1].strip()
            json_data = {}
            json_data['id'] = i
            json_data['relation'] = relation

            tokens, subj_start, subj_end, obj_start, obj_end = get_tokens(sentence)
            json_data['token'] = tokens
            json_data['subj_start'] = subj_start
            json_data['subj_end'] = subj_end
            json_data['obj_start'] = obj_start
            json_data['obj_end'] = obj_end

            json_data['stanford_pos'] = get_pos(tokens)
            ner = get_ner(' '.join(tokens))
            json_data['stanford_ner']  = ner

            json_data['subj_type'], more_than_two = most_common(ner[subj_start:subj_end+1]) 
            if more_than_two:
                more_than_two_list += "more_than_two ner in obj, sentence is : {0}".format(sentence)
                #print("more_than_two ner in subj, sentence is : {0}".format(sentence))
            json_data['obj_type'], more_than_two = most_common(ner[obj_start:obj_end+1]) 
            if more_than_two:
                more_than_two_list += "more_than_two ner in obj, sentence is : {0}".format(sentence)
                #print("more_than_two ner in obj, sentence is : {0}".format(sentence))

            json_data['stanford_deprel'], json_data['stanford_head'] = get_deprel(' '.join(tokens))
            json_datas.append(json_data)
            if i % 50 == 0:
                with open(save_path, 'w') as result_file:
                    json.dump(json_datas, result_file)
            
        except Exception as e:
            error_list += '-'*100
            error_list += data[i] + '\n'
            error_list += str(e) + '\n'
        
    return json_datas

if __name__ == '__main__':
    
    train_data = to_TACRED(train_path, train_json)
    error_list += '-'*40 + ('test') + '-'*40
    more_than_two_list += '-'*40 + ('test') + '-'*40
    test_data = to_TACRED(test_path, test_json)
    with open(MORE_TAHN_TWO, 'w') as f:
        f.write(more_than_two_list)
    with open(ERROR, 'w') as f:
        f.write(error_list)
        
    with open(train_json, 'w') as result_file:
        json.dump(train_data, result_file)
    with open(test_json, 'w') as result_file:
        json.dump(test_data, result_file)