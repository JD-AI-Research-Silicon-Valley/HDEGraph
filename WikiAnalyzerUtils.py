def check_occurance(entity_tokens, support_doc_tokens):
    """
    :param support_tokens_stemmer:
    :param support_tokens:
    :param candidate_answer_tokens:
    :return:
    """
    ###======Exact match=======
    import string
    e_len = len(entity_tokens)
    s_len = len(support_doc_tokens)
    occur_positions = []
    for i in range(s_len - e_len + 1):
        supp_tokens_i = support_doc_tokens[i:(i+e_len)]
        if check_equal_sequence(supp_tokens_i, entity_tokens):
            occur_positions.append((i, i+e_len))

    ###======Punctuation=======
    if len(occur_positions) == 0:
        entity = ' '.join(entity_tokens).strip()
        punc_entity = entity.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation})).strip()
        if len(entity) != len(punc_entity):
            punc_entity_token = punc_entity.split()
            punc_len = len(punc_entity_token)
            for i in range(s_len - punc_len + 1):
                supp_tokens_i = support_doc_tokens[i:(i + punc_len)]
                if check_equal_sequence(supp_tokens_i, punc_entity_token):
                    occur_positions.append((i, i + punc_len))

    ###======Name and Location=======
    if len(occur_positions) == 0 and e_len > 1:
        for i in range(s_len - e_len + 1):
            if(i + e_len + 1 < s_len):
                supp_tokens_i = support_doc_tokens[i:(i + e_len + 1)]
                if entity_tokens[0] == supp_tokens_i[0].lower() \
                        and entity_tokens[-1] == supp_tokens_i[-1].lower() \
                        and supp_tokens_i[0][0] in string.ascii_uppercase \
                        and supp_tokens_i[-1][0] in string.ascii_uppercase:
                    occur_positions.append((i, i + e_len + 1))
                    break

            if (i + e_len + 2 < s_len):
                supp_tokens_i = support_doc_tokens[i:(i + e_len + 2)]
                if entity_tokens[0] == supp_tokens_i[0].lower() \
                        and entity_tokens[-1] == supp_tokens_i[-1].lower() \
                        and supp_tokens_i[0][0] in string.ascii_uppercase \
                        and supp_tokens_i[-1][0] in string.ascii_uppercase:
                    occur_positions.append((i, i + e_len + 2))
                    break

            if (i + e_len + 3 < s_len):
                supp_tokens_i = support_doc_tokens[i:(i + e_len + 3)]
                if entity_tokens[0] == supp_tokens_i[0].lower() \
                        and entity_tokens[-1] == supp_tokens_i[-1].lower() \
                        and supp_tokens_i[0][0] in string.ascii_uppercase \
                        and supp_tokens_i[-1][0] in string.ascii_uppercase:
                    occur_positions.append((i, i + e_len + 3))
                    break

        if len(occur_positions) > 0:
            search_tokens = support_doc_tokens[occur_positions[0][0]:occur_positions[0][1]]
            for i in range(occur_positions[0][0] + 1, s_len - len(search_tokens) + 1):
                supp_tokens_i = support_doc_tokens[i:(i + len(search_tokens))]
                if check_equal_sequence(supp_tokens_i, search_tokens):
                    occur_positions.append((i, i + len(search_tokens)))

    return len(occur_positions) > 0, len(occur_positions), occur_positions

def check_equal_sequence(sup_token_list, entity_token_list):
    """
    Whether two sequence lists are identical
    :param sup_token_list:
    :param entity_token_list:
    :return:
    """
    import operator as op
    for i in range(0, len(entity_token_list)):
        if op.eq(sup_token_list[i].lower(), entity_token_list[i].lower()):
            continue
        else:
            return False
    return True

def Check_Entity_In_Doc(entity_tokens, support_doc_tokens_list):
    """
    :param entity_tokens: list[str]
    :param support_doc_tokens_list: list[list[str]]
    :return: list(doc_position, list((start_position, end_position))
    """
    positions = []
    for doc_idx, doc_tokens in enumerate(support_doc_tokens_list):
        flag, count, in_doc_positions = check_occurance(entity_tokens, doc_tokens)
        if flag:
            for pos in in_doc_positions:
                positions.append([doc_idx, pos[0], pos[1]])
    return len(positions) > 0, positions