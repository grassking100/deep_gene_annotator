class DNA_CodeException(Exception):
    pass
class DNA_SeqException(Exception):
    pass
#convert DNA code to one hot encoding
def code2vec(code):
    target=['A','T','C','G']
    a=[1,0,0,0]
    t=[0,1,0,0]
    c=[0,0,1,0]
    g=[0,0,0,1]
    vec=[a,t,c,g]
    for i in range(len(target)):
        if code.upper()==target[i]:
            return vec[i]
    else:
        raise DNA_CodeException(str(code)+' is not in space')
#convert one hot encoding to DNA code
def vec2code(vector):
    a=[1,0,0,0]
    t=[0,1,0,0]
    c=[0,0,1,0]
    g=[0,0,0,1]
    target=[a,t,c,g]
    code=['A','T','C','G']
    for i in range(len(target)):
        if vector==target[i]:
            return code[i]
    raise DNA_CodeException(str(vector)+' is not in space')
#convert DNA sequence to one hot encoding sequence
def codes2vec(codes):
    code_list=list(codes)
    arr=[]
    for code in code_list:
        try:
            arr.append(code2vec(code))
        except DNA_CodeException as e:
            raise DNA_SeqException('Sequence has invalid code in it')
    return arr
#convert one hot encoding sequence to DNA sequence
def vec2codes(vector):
    vectors=list(vector)
    arr=[]
    for v in vectors:
        try:
            arr.append(vec2code(v))
            
        except DNA_CodeException as e:
            raise DNA_SeqException('Sequence vector has invalid vector in it')
    return arr