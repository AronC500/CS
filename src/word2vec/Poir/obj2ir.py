import os
import re
import string
import argparse

# Attribute
OBJ_ATTR = r"/[A-Za-z0-9\+\-_.#]+"

# Type: Atomic and Composite
OBJ_TYPE_LIST = r"\[.*?\]"                         # LIST
# OBJ_TYPE_STR1 = r"\(.*?(?<!\\)\)"                  # STR(Benign)
OBJ_TYPE_STR1 = r"\(.*?\)"                         # STR 
OBJ_TYPE_STR2 = r"<[A-Z0-9]*>"                     # STR(ID)
OBJ_TYPE_REF = r"\d+\s+\d+\s+R"                    # REF
OBJ_TYPE_NUM1 = r"[-+]?\d*\.\d+"                   # NUM
OBJ_TYPE_NUM2 = r"[-+]?\d+"                        
OBJ_TYPE_NAME = r"/[A-Za-z0-9\+\-_.#]+"              # NAME
OBJ_TYPE_BOOL = r"true|false"                      # BOOL
OBJ_TYPE_NULL = r"null"                            # NULL
OBJ_TYPE_STREAM = r"stream_endstream"              # STREAM
OBJ_TYPE_DICT1 = r'<<'                             # DICT
OBJ_TYPE_DICT2 = r'>>'
ALL_TYPES = [OBJ_TYPE_LIST, OBJ_TYPE_STR1, OBJ_TYPE_STR2, OBJ_TYPE_REF, OBJ_TYPE_NUM1, OBJ_TYPE_NUM2, OBJ_TYPE_NAME, OBJ_TYPE_BOOL, OBJ_TYPE_NULL, OBJ_TYPE_STREAM, OBJ_TYPE_DICT1, OBJ_TYPE_DICT2]



def get_obj_type(match, ptr):
    min_index = len(match)
    index = 0
    for i, pattern in enumerate(ALL_TYPES):
        value = re.search(pattern, match[ptr:], re.DOTALL)
        if value:
            if value.start() < min_index:
                min_index = value.start()
                index = i
    return ALL_TYPES[index]



def get_atomic_type(value):
    if value == OBJ_TYPE_STR1 or value == OBJ_TYPE_STR2:
        return 'STR'
    elif value == OBJ_TYPE_REF:
        return 'REF'
    elif value == OBJ_TYPE_NUM1 or value == OBJ_TYPE_NUM2:
        return 'NUM'
    elif value == OBJ_TYPE_NAME:
        return 'NAME'
    elif value == OBJ_TYPE_BOOL:
        return 'BOOL'
    elif value == OBJ_TYPE_NULL:
        return 'NULL'
    else:
        return 'NOTFOUNDTYPE'
    


def get_list(text):
    index = text.find('[')
    list_len = index
    #
    match = []
    match.append(text[index])
    index += 1

  
    stack = 1
    while stack != 0 and index < len(text):
        if text[index] == '[' and text[index - 1] != '/' and text[index - 1] != '\\':
            stack += 1
        elif text[index] == ']' and text[index - 1] != '/' and text[index - 1] != '\\':
            stack -= 1
        match.append(text[index])
        index += 1
    list_str = ''.join(match)
    list_len += len(list_str)
    return [list_str, list_len]



def get_str(text):
    index = text.find('(')
    str_len = index
    
    match = []
    match.append(text[index])
    index += 1


    stack = 1
    while stack != 0 and index < len(text):
        if text[index] == '(' and text[index - 1] != '/' and text[index - 1] != '\\':
            stack += 1
        elif text[index] == ')' and text[index - 1] != '/' and text[index - 1] != '\\':
            stack -= 1
        match.append(text[index])
        index += 1
    s_str = ''.join(match)
    str_len += len(s_str)
    return [s_str, str_len]



def llist_to_list(text):
   
    stack = 0
    new_text = list(text)
    for i in range(len(new_text)):
        if i < len(new_text) - 1 and new_text[i:i+2] == ['<','<']:
            stack += 1
        if i < len(new_text) - 1 and new_text[i:i+2] == ['>','>']:
            stack -= 1
        if (not stack) and (new_text[i] == '[' or text[i] == ']'):
            new_text[i] = ' '
    return ''.join(new_text)



def parse_list(text):
    # 
    text = llist_to_list(text)

    # 
    llist_values = [['DICT'], ['STR'], ['NAME'], ['BOOL'], ['REF'], ['NUM']]

    obj_type_dict = r"<<.*?>>"
    patterns = [obj_type_dict, OBJ_TYPE_STR1, OBJ_TYPE_STR2, OBJ_TYPE_NAME, OBJ_TYPE_BOOL, OBJ_TYPE_REF, OBJ_TYPE_NUM1, OBJ_TYPE_NUM2]
    for i, pattern in enumerate(patterns):
        matches = re.findall(pattern, text, re.DOTALL)
        text = re.sub(pattern, '', text, flags=re.DOTALL)
        for match in matches:
            if pattern == obj_type_dict:
                # 
                match = re.sub(r'stream.*?endstream', 'stream_endstream', match, flags=re.DOTALL)
                match = match.strip()
                match = match.replace('\n',' ')
                llist_values[0].append(match)
            elif pattern == OBJ_TYPE_STR1 or pattern == OBJ_TYPE_STR2:
                # 
                llist_values[1].append(match)
            elif pattern == OBJ_TYPE_NUM1 or pattern == OBJ_TYPE_NUM2:
                # 
                llist_values[-1].append(match)
            else:
                llist_values[i-1].append(match)


    new_list = [element for element in llist_values if len(element) != 1]
    llist_values = new_list


    final_dict_pre = []   
    final_type = []       
    final_values = []
    if len(llist_values) == 1:
        # 
        if llist_values[0][0] == 'DICT':
            # 
            final_dict_pre.append('')
            final_type.append('DICT')
            final_values.append('<Blank>')
            for i in llist_values[0][1:]:
                temp_id = 'temp_id'
                temp_ir = get_objIR(temp_id, i)
                temp_ir = [element for element in temp_ir if element != 'temp_id, ' and element != '\n']
                for k in range(0, len(temp_ir), 3):
                    final_dict_pre.append(temp_ir[k])
                    final_type.append(temp_ir[k+1])
                    final_values.append(temp_ir[k+2])

        elif llist_values[0][0] == 'REF':
            # 
            final_dict_pre.append('')
            final_type.append('REF_LIST')
            final_values.append(len(llist_values[0][1:]))
            for i in llist_values[0][1:]:
                final_dict_pre.append('')
                final_type.append('REF')
                # 
                temp_matches = re.findall(r'\d+', i)
                final_values.append(''.join(temp_matches))
        else:
            final_dict_pre.append('')
            final_type.append(llist_values[0][0] + '_LIST')
            final_values.append(len(llist_values[0][1:]))

    else:
        # 
        final_dict_pre.append('')
        final_type.append('MIX_LIST')
        final_values.append(sum(len(i) - 1 for i in llist_values))
        if any(sublist[0] == 'DICT' for sublist in llist_values):
            # 
            for sublist in llist_values:
                if sublist[0] == 'DICT':
                    final_dict_pre.append('')
                    final_type.append('DICT')
                    final_values.append('<Blank>')
                    for i in sublist[1:]:
                        temp_id = 'temp_id'
                        temp_ir = get_objIR(temp_id, i)
                        temp_ir = [element for element in temp_ir if element != 'temp_id, ' and element != '\n']
                        for k in range(0, len(temp_ir), 3):
                            final_dict_pre.append(temp_ir[k])
                            final_type.append(temp_ir[k+1])
                            final_values.append(temp_ir[k+2])
                    break

        if any(sublist[0] == 'REF' for sublist in llist_values):
            # 
            for sublist in llist_values:
                if sublist[0] == 'REF':
                    for i in sublist[1:]:
                        final_dict_pre.append('')
                        final_type.append('REF')
                        # 
                        temp_matches = re.findall(r'\d+', i)
                        final_values.append(''.join(temp_matches))
                    break
        
    return [final_dict_pre, final_type, final_values]


def get_objIR(id, match):
    # 
    if match == '<<>>' or match == '<< >>' or match == '<<  >>' \
        or match == '' or match == ' ' or match == '  ' or match == 'null':
        return [id + ', ', '<Blank>, ', 'NULL, ', '<Blank>', '\n']
    
    # 
    if match[0:2] != '<<': 
        if match.startswith("[") and match.endswith("]"):
            match = "<< /SINGLELIST " + match + " >>"
        elif match.startswith("/"):
            match = "<< /SINGLENAME " + match + " >>"
        elif match.startswith("(") and match.endswith(")"):
            match = "<< /SINGLESTR " + match + " >>"
            # return [id + ', ', '/SINGLESTR, ', 'STR, ', match, '\n']
        elif re.search(OBJ_TYPE_NUM1, match) or re.search(OBJ_TYPE_NUM2, match):
            match = "<< /SINGLENUM " + match + " >>"
        else:
            # match = "<< /SINGLE" + match + ">>"
            print(match)

    # 
    ir = []
    ptr = 2
    dict_pre = []
    dict_stack = 1
    while ptr < len(match):
        ir.append(id + ', ')

        # 
        attr = re.search(OBJ_ATTR, match[ptr:])
        ir.append(''.join(dict_pre) + attr.group() + ', ')
        ptr += attr.end()

       
        select_type = get_obj_type(match, ptr)
        if select_type == OBJ_TYPE_DICT1:
            # 
            value_match = re.search(OBJ_TYPE_DICT1, match[ptr:])
            dict_pre.append(attr.group())
            ir.append('DICT, ')
            ir.append('<Blank>')
            ir.append('\n')
            ptr += value_match.end()
        elif select_type == OBJ_TYPE_LIST:
            #
            [value_match, list_len] = get_list(match[ptr:])
            [final_dict_pre, final_type, final_values] = parse_list(value_match)

            temp_attribute = ir[-1]
            ir = ir + [final_type[0] + ', '] + [final_values[0]] + ['\n']
            if len(final_dict_pre) != 1:
                # 
                for i in range(1, len(final_dict_pre)):
                    if final_dict_pre[i] != '':
                        ir = ir + [id + ', '] + [temp_attribute[:-2] + final_dict_pre[i]] + [final_type[i]] + [final_values[i]] + ['\n']
                    else:
                        ir = ir + [id + ', '] + [temp_attribute] + [final_type[i] + ', '] + [final_values[i]] + ['\n']
            ptr += list_len  
        else:
            # 
            atomic_type = get_atomic_type(select_type)
            ir.append(atomic_type + ', ')

            if atomic_type == 'REF':
                # 
                value_match = re.search(select_type, match[ptr:], re.DOTALL)
                temp_matches = re.findall(r'\d+', value_match.group())
                ir.append(temp_matches[0] + temp_matches[1])
                ir.append('\n')
                ptr += value_match.end()
            elif atomic_type == 'STR':
                #  
                [value_match, str_len] = get_str(match[ptr:])
                ir.append(value_match)
                ir.append('\n')
                ptr += str_len
            else:
                value_match = re.search(select_type, match[ptr:], re.DOTALL)
                ir.append(value_match.group())
                ir.append('\n')
                ptr += value_match.end()

        while (get_obj_type(match, ptr) == OBJ_TYPE_DICT2):
            #  
            dict_stack -= 1
            if dict_pre != []:
                dict_pre.pop()
            end_match = re.search(OBJ_TYPE_DICT2, match[ptr:])
            ptr += end_match.end()
        if re.search(OBJ_TYPE_STREAM, match):
            #  
            ir = [id + ', '] + ['<Blank>, '] + ['STREAM, '] + ['<Blank>'] + ['\n'] + ir
            #  
            stream_match = re.search(OBJ_TYPE_STREAM, match)
            #  
            match = match[:stream_match.start()]
            match = match.strip()
        if dict_stack == 0:
            break
    return ir


#  
def output_sentences(pdf_file_path, outfn):
    with open(pdf_file_path, 'r', encoding='ISO 8859-2') as file:
        file_contents = file.read()

    #  
    file_contents = ''.join(char for char in file_contents if char in string.printable)

    #  
    pattern = r'(\d+) (\d+) obj(.*?)endobj'
    matches = re.findall(pattern, file_contents, re.DOTALL)

    #  
    sorted_matches = sorted(matches, key=lambda match: int(match[0] + match[1]))

    obj_IRs = []
    for sorted_match in sorted_matches:
        #   
        temp_match = re.sub(r'stream.*?endstream', 'stream_endstream', sorted_match[2], flags=re.DOTALL)
        temp_match = temp_match.strip()
        temp_match = temp_match.replace('\n',' ')
        temp_ID = sorted_match[0] + sorted_match[1]
        # if temp_ID == '50':
        #     print('be care')

        #  
        try:
            temp_IR = get_objIR(temp_ID, temp_match)
            obj_IRs = obj_IRs + temp_IR
        except:
            print(f"{pdf_file_path} have error obj: {temp_ID}")

    with open(outfn, 'w', encoding='utf-8', newline='') as f:
        obj_IRs = [str(item) if isinstance(item, int) else item for item in obj_IRs]
        f.write(''.join(obj_IRs))
        f.write('\n')


#  
def preprocess_pdfs_dir(pdfs_dir, out_dir):
    pdf_fns = os.listdir(pdfs_dir)
    for fn in pdf_fns:
        if fn == 'sample':
            print('here')
        file_path = os.path.join(pdfs_dir, fn)
        #  
        if '.pdf' in fn:
            outfn = fn.split('.pdf')[0]
        else:
            outfn = fn
        out_path = os.path.join(out_dir, outfn)
        output_sentences(file_path, out_path)

# todo
# error handling
        
    
if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description='Convert PDF Object to IR')
    parser.add_argument('-i', required=True, dest='input_files_dir', action='store', help='Input PDF files dir' )
    parser.add_argument('-o', required=True, dest='out_dir', action='store', help='Output PDFObj IR files dir' )

    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    preprocess_pdfs_dir(args.input_files_dir, args.out_dir)