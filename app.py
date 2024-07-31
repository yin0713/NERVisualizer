from io import StringIO
import streamlit as st
import spacy_streamlit
import json

st.set_page_config(page_title="visualize-ner", page_icon=":mushroom:")
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

test_true = []
test_pred = []
train_true = []
train_pred = []

try:
    expander0000000 = st.expander("input the entity label definition -->must")
    labels = expander0000000.text_area(label='', value='', placeholder="input the label and seperate each label with any whitespace")
    labels = list(sorted(list(set([i.strip() for i in labels.split() if i.strip()]))))
    expander000000 = st.expander("upload test or train data  -->must")
    tab_test, tab_train = expander000000.tabs(['test', 'train'])
    with tab_test:
        true_file_test = st.file_uploader("choose a test true file", accept_multiple_files=False)
        pred_file_test = st.file_uploader("choose a test pred file", accept_multiple_files=False)
        st.json([{'id': 1, "text": "", "entities": [{"start_idx": 0, "end_idx": 2, "entity": "", "type": ""}]}], expanded=False)

    with tab_train:
        true_file_train = st.file_uploader("choose a train true file", accept_multiple_files=False)
        pred_file_train = st.file_uploader("choose a train pred file", accept_multiple_files=False)
        st.json([{'id': 1, "text": "", "entities": [{"start_idx": 0, "end_idx": 2, "entity": "", "type": ""}]}], expanded=False)

    if true_file_test and pred_file_test:
        test_true = json.loads(StringIO(true_file_test.getvalue().decode("utf-8")).read())
        test_pred = json.loads(StringIO(pred_file_test.getvalue().decode("utf-8")).read())
        assert len(test_true) == len(test_pred)

    if true_file_train and pred_file_train:
        train_true = json.loads(StringIO(true_file_train.getvalue().decode("utf-8")).read())
        train_pred = json.loads(StringIO(pred_file_train.getvalue().decode("utf-8")).read())
        assert len(train_true) == len(train_pred)
except:
    st.write('上传的true和pred数据长度不一致！')
    st.stop()

for record in test_true:
    record['trainortest'] = 'test'
for record in test_pred:
    record['trainortest'] = 'test'
for record in train_true:
    record['trainortest'] = 'train'
for record in train_pred:
    record['trainortest'] = 'train'

trues = test_true + train_true
preds = test_pred + train_pred
all_records = [trues, preds]

#条件1
expander0 = st.expander("select all or train or test data to show")
trainortest_select = expander0.selectbox("", options=['all', 'train', 'test'], index=2)

#条件2
expander000 = st.expander("filter dataset by sample text")
searchtext = expander000.text_input('', '')

#条件3
expander000 = st.expander("filter dataset by entity text")
with expander000:
    col0, col1, col2, col4 = st.columns([2, 4, 2, 2])
    searchentity = col1.text_input('', '', key=1)
    equalorin = col2.selectbox('', ['equal', 'in'], index=1)

#条件4
expander = st.expander("select which entity to evaluate")
entity_to_evaluate = expander.multiselect('', options=labels, default=labels)

#条件5
expander1 = st.expander("Select all_cases or good_cases or bad_cases to show")
label_select = expander1.selectbox("", options=['all_cases', 'good_cases', 'bad_cases'], index=2)


def filter_with_searchteext(to_filter, searchtext):
    filtered = [[], []]
    for true, pred in zip(to_filter[0], to_filter[1]):
        text = true['text']
        if searchtext in text:
            filtered[0].append(true)
            filtered[1].append(pred)
    return filtered


def filter_with_searchentity(to_filter, searchentity, equalorin):
    filtered = [[], []]
    for true, pred in zip(to_filter[0], to_filter[1]):
        entities_true = true['entities']
        entities_pred = pred['entities']
        flag = 0
        for entity in entities_true:
            entity_text = entity['entity']
            if equalorin == 'equal':
                if searchentity == entity_text:
                    filtered[0].append(true)
                    filtered[1].append(pred)
                    flag = 1
                    break
            elif equalorin == 'in':
                if searchentity in entity_text:
                    filtered[0].append(true)
                    filtered[1].append(pred)
                    flag = 1
                    break
        if flag:
            continue
        for entity in entities_pred:
            entity_text = entity['entity']
            if equalorin == 'equal':
                if searchentity == entity_text:
                    filtered[0].append(true)
                    filtered[1].append(pred)
                    break
            elif equalorin == 'in':
                if searchentity in entity_text:
                    filtered[0].append(true)
                    filtered[1].append(pred)
                    break
    return filtered


def filter_with_trainortest_select(to_filter, trainortest_select):
    if trainortest_select == 'all':
        return to_filter
    filtered = [[], []]
    for true, pred in zip(to_filter[0], to_filter[1]):
        trainortest = true['trainortest']
        if trainortest_select == trainortest:
            filtered[0].append(true)
            filtered[1].append(pred)
    return filtered


def entity_filter(record, entity_to_evaluate):
    text = record['text']
    trainortest = record['trainortest']
    entities = []
    for entity in record['entities']:
        if entity['type'] in entity_to_evaluate:
            entities.append(entity)
    return {'text': text, 'entities': entities, 'trainortest': trainortest}


def filter_with_entity_to_evaluate(to_filter, entity_to_evaluate):
    filtered = [[], []]
    for true, pred in zip(to_filter[0], to_filter[1]):
        filtered[0].append(entity_filter(true, entity_to_evaluate))
        filtered[1].append(entity_filter(pred, entity_to_evaluate))
    return filtered


filtered = filter_with_trainortest_select(to_filter=all_records, trainortest_select=trainortest_select)
filtered = filter_with_searchteext(to_filter=filtered, searchtext=searchtext)
filtered = filter_with_searchentity(to_filter=filtered, searchentity=searchentity, equalorin=equalorin)
filtered = filter_with_entity_to_evaluate(to_filter=filtered, entity_to_evaluate=entity_to_evaluate)
trues = filtered[0]
preds = filtered[1]


#origin format  {'text':text, 'entities':[{'entity':entity, 'start_idx':start_idx, 'end_idx':end_idx, 'type':type}]}
#spacy-streamlit format {'text':text, 'ents':[{'entity':entity, 'start':start, 'end':end, 'label':label}]}
def convert_spacy_streamlit(records, title=False):
    spacy_streamlit_format = []
    for idx, record in enumerate(records):
        # print(record)
        text = record['text']
        trainortest = record['trainortest']
        entities = record['entities']
        ents = []
        for ent in entities:
            entity = ent['entity']
            start = ent['start_idx']
            end = ent['end_idx']
            label = ent['type']
            ents.append({'entity': entity, 'start': start, 'end': end, 'label': label})
        if title:
            spacy_streamlit_format.append({'title': trainortest, 'text': text, 'ents': ents})
        else:
            spacy_streamlit_format.append({'text': text, 'ents': ents})
    return spacy_streamlit_format


trues = convert_spacy_streamlit(trues, title=True)
preds = convert_spacy_streamlit(preds)


def is_consistent(trues, preds):
    for true in trues:
        if true not in preds:
            return False
    for pred in preds:
        if pred not in trues:
            return False
    return True


docs = []
for t, p in zip(trues, preds):
    if label_select == 'all_cases':
        docs.extend([t, p])
    elif label_select == 'bad_cases' and not is_consistent(t['ents'], p['ents']):
        docs.extend([t, p])
    elif label_select == 'good_cases' and is_consistent(t['ents'], p['ents']):
        docs.extend([t, p])

if entity_to_evaluate:
    spacy_streamlit.visualize_ner(
        docs,
        manual=True,
        show_table=False,
        labels=entity_to_evaluate,
        title='',
    )
