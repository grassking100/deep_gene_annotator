#convert DNA code to one hot encoding
def code2vec(code,safe=False):
    target=['A','T','C','G']
    a=[1,0,0,0]
    t=[0,1,0,0]
    c=[0,0,1,0]
    g=[0,0,0,1]
    vec=[a,t,c,g]
    '''
    A=1000
    T=0100
    C=0010
    G=0001
    '''
    for i in range(len(target)):
        if code.upper()==target[i]:
            return vec[i]
    #check if there is any invalid code 
    if safe:
        assert False,(str)(code)+' is not in space'
    else:
        return None
#convert one hot encoding to DNA code
def vec2code(vector,safe=False):
    a=[1,0,0,0]
    t=[0,1,0,0]
    c=[0,0,1,0]
    g=[0,0,0,1]
    target=[a,t,c,g]
    code=['A','T','C','G']
    for i in range(len(target)):
        if vector==target[i]:
            return code[i]
    if safe:
        assert False,(str)(vector)+' is not in space'
    return None
#convert DNA sequence to one hot encoding sequence
def codes2vec(code,safe=False):
    codes=list(code)
    arr=[]
    for c in code:
        vec=code2vec(c,safe)
        if vec is not None:
            arr.append(vec)
        else:
            return None
    return arr
#convert one hot encoding sequence to DNA sequence
def vec2codes(vector,safe=False):
    vectors=list(vector)
    arr=[]
    for v in vectors:
        code=vec2code(v,safe)
        if code is not None:
            arr.append(code)
        else:
            return None
    return arr