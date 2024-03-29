import collections
import json

from tqdm import tqdm

from ..tools import official_tokenization as tokenization

SPIECE_UNDERLINE = '▁'
DEBUG = True


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def read_cmrc_examples(input_file, two_level_embeddings, has_labels=True):
    train_data = [json.loads(line) for line in open(input_file, 'r')]

    def _is_chinese_char(cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def is_fuhao(c):
        if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
                or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
                or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
                or c == '‘' or c == '’':
            return True
        return False

    def _tokenize_chinese_chars(text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if _is_chinese_char(cp) or is_fuhao(char):
                if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                    output.append(SPIECE_UNDERLINE)
                output.append(char)
                output.append(SPIECE_UNDERLINE)
            else:
                output.append(char)
        return "".join(output)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
            return True
        return False

    # to examples
    examples = []
    mis_match = 0
    for query in tqdm(train_data):
        context = query['context']
        qid = query['id']
        answers = query['answers']
        
        if two_level_embeddings:
            # Remove strange characters
            context = context.replace('\u200b', '')
            context = context.replace(u'\xa0', u'')
            # Adjust answer position accordingly
            for i, ans in enumerate(answers):
                ans_text = ans['text']
                ans_start = ans['answer_start']
                if ans_text != context[ans_start:ans_start + len(ans_text)]:
                    lo = None
                    for offset in range(-3, 4):
                        lo = ans_start + offset
                        if context[lo:lo+len(ans_text)] == ans_text:
                            answers[i]['answer_start'] = lo
                            break


        context_chs = _tokenize_chinese_chars(context)
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in context_chs:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            if c != SPIECE_UNDERLINE:
                char_to_word_offset.append(len(doc_tokens) - 1)

        # For training, learn to output first answer only.
        ques_text = query['question']
        ans_text = answers[0]['text']

        start_position_final = None
        end_position_final = None

        if has_labels:  # This is always true for now
            count_i = 0
            start_position = answers[0]['answer_start']

            end_position = start_position + len(ans_text) - 1
            repeat_limit = 3
            while context[start_position:end_position + 1] != ans_text and count_i < repeat_limit:
                start_position -= 1
                end_position -= 1
                count_i += 1

            while context[start_position] == " " or context[start_position] == "\t" or \
                    context[start_position] == "\r" or context[start_position] == "\n":
                start_position += 1

            start_position_final = char_to_word_offset[start_position]
            end_position_final = char_to_word_offset[end_position]

            if doc_tokens[start_position_final] in {"。", "，", "：", ":", ".", ","}:
                start_position_final += 1

            actual_text = "".join(doc_tokens[start_position_final:(end_position_final + 1)])
            cleaned_answer_text = "".join(tokenization.whitespace_tokenize(ans_text))

            if actual_text != cleaned_answer_text:
                print(actual_text, 'V.S', cleaned_answer_text)
                mis_match += 1

        examples.append({
            'doc_tokens': doc_tokens,
            'orig_answer_text': ans_text,
            'qid': qid,
            'question': ques_text,
            'answer': ans_text,
            'start_position': start_position_final,
            'end_position': end_position_final})
    return examples, mis_match


def get_subchar_pos(tokens, subchars):
    '''
    Return starting index of each subchar in tokens.
    NOTE: This assumes that the concatenation of tokens is equal to the 
    concatenation of subchars.

    Example:
    >>> Input:
    >>> subchars  = ['jin+', 'ti', 'an+', 'ti', 'an+', 'qi+', 'hen+', 'hao+']
    >>> tokens    = ['jin', '+', 'tian+', 'tian+qi', '+', 'hen+hao+']
    >>> token_pos = [0, 2, 2, 3, 3, 3, 5, 5]
    '''
    if ''.join(tokens) != ''.join(subchars):
        print(tokens)
        print(subchars)
        print('\n\n')
    assert ''.join(tokens) == ''.join(subchars)
    pos = [None] * len(subchars)
    len_t = 0
    len_s = 0
    j = -1  # idx of last token that was added to len_t
    for i, subchar in enumerate(subchars):
        while len_t <= len_s:
            j += 1
            len_t += len(tokens[j])
        pos[i] = j
        len_s += len(subchar)
    return pos


def convert_examples_to_features_twolevel(
    examples,
    tokenizer,
    max_seq_length=512,
    max_query_length=64,
    doc_stride=128,
     include_long_tokens=False):

    is_training = True  # TODO: Just remove this parameter altogether.

    features = []
    unique_id = 1000000000
    for (example_index, example) in enumerate(tqdm(examples)):
        question = example['question']
        query_tokens = tokenizer.tokenize(question)
        query_tokens = query_tokens[:max_query_length]  # Truncate too long questions

        doc_tokens = example['doc_tokens']

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        long_tokens = tokenizer.tokenize(''.join(doc_tokens))
        
        # Remove weird chars
        for i in range(len(long_tokens)):
            for c in ['\u2581', '\u3000', '\u2009']:
                long_tokens[i] = long_tokens[i].replace(c, '')
        
        try:
            doc_tok_to_long_index = get_subchar_pos(long_tokens, all_doc_tokens)
        except Exception as e:
            print('Getting subchar pos failed')
            print(doc_tokens)
            print(long_tokens)
            print(all_doc_tokens)

            print(''.join(long_tokens))
            print(''.join(all_doc_tokens))
   
            raise e
            # a = tokenizer.tokenize('俄罗斯圣彼得堡的模特儿。')
            # s = '圣彼得堡的'
            # b = tokenizer.tokenize(s)
            # for i in range(len(b)):
            # 	b[i] = b[i].replace('▁', '')
            # bs = sum([tokenizer.tokenize(c) for c in s], [])
            # # print(a)
            # print(b)
            # print(bs)
            # # print(''.join(a))
            # a = ''.join(b)
            # b = ''.join(bs)
            # for i in range(len(a)):
            # 	print(i, a[i], ord(a[i]), b[i], ord(b[i]))
            exit()

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = orig_to_tok_index[example['start_position']]  # 原来token到新token的映射，这是新token的起点
            if example['end_position'] < len(example['doc_tokens']) - 1:
                tok_end_position = orig_to_tok_index[example['end_position'] + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example['orig_answer_text'])

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = orig_to_tok_index[example['start_position']]  # 原来token到新token的映射，这是新token的起点
            if example['end_position'] < len(example['doc_tokens']) - 1:
                tok_end_position = orig_to_tok_index[example['end_position'] + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example['orig_answer_text'])

        # The -3 accounts for [CLS], [SEP] and [SEP]
        # max_tokens_for_doc = max_seq_length - len(query_subchars) - 3
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        doc_spans = []
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        # Each doc span may be a new feature
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            #
            # Feature format:
            #    tokens      = [CLS] + <query tokens> + [SEP] + <all_doc_token> + [SEP]
            #    segment_id  =   0   0  ...               0   0   1   1   ...    1  1
            #    long_tokens = ...
            #    subchar_pos = ...
            #
            cls_token = '[CLS]'
            sep_token = '[SEP]'
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            tokens.append(cls_token)
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append(sep_token)
            segment_ids.append(0)
            token_to_long_index = [-1] * len(tokens)  # Map index of token to long_tokens

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
                
                token_to_long_index.append(doc_tok_to_long_index[split_token_index])
            tokens.append("[SEP]")
            segment_ids.append(1)
            token_to_long_index.append(-1)


            # Extract tokens that overlap with tokens in this span
            long_index_lo = doc_tok_to_long_index[doc_span.start]
            # long_index_hi = doc_tok_to_long_index[doc_span.start + doc_span.length]
            this_long_tokens = long_tokens[long_index_lo:]  		# Excessive long tokens on the right end doesn't matter
            this_long_tokens = this_long_tokens[:max_seq_length]  	# But can't be too long
            # Tweak index mapping of tokens accordingly
            for i in range(len(token_to_long_index)):
                if token_to_long_index[i] != -1:
                    token_to_long_index[i] -= long_index_lo

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            token_ids = tokenizer.convert_tokens_to_ids(long_tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Turn token_to_long_index into left and right pos
            pos_left = []
            pos_right = []
            for i, index in enumerate(token_to_long_index):
                if index == -1:
                    pos_left.append(-1)
                    pos_right.append(-1)
                else:
                    pos_left.append(index)
                    # Make sure right index is at least greater than left index. 
                    # next_index will be -1 for the rightmost token, and next_index will 
                    # be equal to this index when next token map to same long token.
                    next_index = token_to_long_index[i+1]
                    pos_right.append(max(index + 1, next_index))

            # subchar_pos.append(len(token_ids))
            # pos_left = [subchar_pos[i] for i in range(len(subchar_pos) - 1)]
            # pos_right = [subchar_pos[i+1] for i in range(len(subchar_pos) - 1)]

            # # When multiple subchars map to one token, corresponding pos_left and pos_right will be equal
            # # except for the rightmost subchar. We add one to make sure the token embedding will be added
            # # to all subchars.
            # for i in range(len(pos_left) - 1):
            # 	if pos_left[i] == pos_right[i]:
            # 		pos_right[i] += 1

            # Zero-pad up to the sequence length.
            padding_len = max_seq_length - len(input_ids)
            token_padding_len = max_seq_length - len(token_ids)

            input_ids += [0] * padding_len
            pos_left += [-1] * padding_len  # Doesn't matter
            pos_right += [-1] * padding_len
            input_mask += [0] * padding_len
            segment_ids += [0] * padding_len
            token_ids += [0] * token_padding_len

            # There might be more tokens than subchars, so we need to truncate it.
            token_ids = token_ids[:max_seq_length]

            if len(token_ids) != max_seq_length:
                print(len(token_ids))
                print(max_seq_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(token_ids) == max_seq_length
            assert len(pos_left) == len(pos_right) == max_seq_length

            start_position = None
            end_position = None
            is_training = True
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                if tok_start_position == -1 and tok_end_position == -1:
                    start_position = 0  # 问题本来没答案，0是[CLS]的位子
                    end_position = 0
                else:  # 如果原本是有答案的，那么去除没有答案的feature
                    out_of_span = False
                    doc_start = doc_span.start  # 映射回原文的起点和终点
                    doc_end = doc_span.start + doc_span.length - 1

                    if not (tok_start_position >= doc_start and tok_end_position <= doc_end):  # 该划窗没答案作为无答案增强
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

            feature = {'unique_id': unique_id,
                       'example_index': example_index,
                       'doc_span_index': doc_span_index,
                    #    'text': text,
                    #    'subchars': subchars,
                       'tokens': tokens,
                       'token_to_orig_map': token_to_orig_map,
                       'token_is_max_context': token_is_max_context,
                       'input_ids': input_ids,
                       'input_mask': input_mask,
                       'segment_ids': segment_ids,
                       'start_position': start_position,
                       'end_position': end_position,

                       # For token embeddings
                       'token_ids': token_ids,
                       'pos_left': pos_left,
                       'pos_right': pos_right,
                    #    'index_subchar_to_char': index_subchar_to_char,
                    #    'subchar_is_max_context': subchar_is_max_context,
                       }
            if include_long_tokens:
                feature['long_tokens'] = long_tokens
            # print(token_ids)
            # exit()
            features.append(feature)
            unique_id += 1

    return features


def convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length: int=512,
    max_query_length: int=64,
    doc_stride: int=128,
    two_level_embeddings: bool=False,
    include_long_tokens: bool=False,
    is_training: bool=True,         # NOTE: This might not work if set to False
    avg_char_tokens: bool=None,     # Will be ignored
    ):
    if two_level_embeddings:
        return convert_examples_to_features_twolevel(
            examples,
            tokenizer,
            max_seq_length=max_seq_length,
            max_query_length=max_query_length,
            doc_stride=doc_stride,
            include_long_tokens=include_long_tokens,
        )

    features = []
    unique_id = 1000000000
    for (ex_idx, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example['question'])
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example['doc_tokens']):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = orig_to_tok_index[example['start_position']]  # 原来token到新token的映射，这是新token的起点
            if example['end_position'] < len(example['doc_tokens']) - 1:
                tok_end_position = orig_to_tok_index[example['end_position'] + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example['orig_answer_text'])

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # Iterate different spans of the context to generate multiple features.
        doc_spans = []
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                if tok_start_position == -1 and tok_end_position == -1:
                    start_position = 0  # 问题本来没答案，0是[CLS]的位子
                    end_position = 0
                else:  # 如果原本是有答案的，那么去除没有答案的feature
                    out_of_span = False
                    doc_start = doc_span.start  # 映射回原文的起点和终点
                    doc_end = doc_span.start + doc_span.length - 1

                    if not (tok_start_position >= doc_start and tok_end_position <= doc_end):  # 该划窗没答案作为无答案增强
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

            features.append({'unique_id': unique_id,
                             'example_index': ex_idx,
                             'doc_span_index': doc_span_index,
                             'tokens': tokens,
                             'token_to_orig_map': token_to_orig_map,
                             'token_is_max_context': token_is_max_context,
                             'input_ids': input_ids,
                             'input_mask': input_mask,
                             'segment_ids': segment_ids,
                             'start_position': start_position,
                             'end_position': end_position})
            unique_id += 1
            
            # Log progress
            if ex_idx % 2000 == 0:
                print(f'Processed {ex_idx} / {len(examples)} examples')

    return features
 
 
