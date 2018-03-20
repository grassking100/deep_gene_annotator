import unittest
from . import code2vec,vec2code,codes2vec,vec2codes,seqs2dnn_data
class TestDNAVector(unittest.TestCase):
    def test_code2vec(self):
        codes=['A','T','C','G','a','t','c','g']
        vectors=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        for index in range(len(codes)):
            self.assertEqual(vectors[index%4],code2vec(codes[index]))
    def test_code2vec_unsafe(self):
        codes=['i']
        self.assertEqual(None,code2vec(codes[0],False))
    def test_code2vec_safe(self):
        codes=['i']
        with self.assertRaises(Exception) as context:
            code2vec(codes[0])
        self.assertTrue(codes[0]+" is not in space"==str(context.exception))
    def test_vec2code(self):
        codes=['A','T','C','G']
        vectors=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        for index in range(len(vectors)):
            self.assertEqual(codes[index],vec2code(vectors[index]))
            
    def test_vec2code_safe(self):
        invalid_vec=[1,1,0,0]
        with self.assertRaises(Exception) as context:
            vec2code(invalid_vec)
        self.assertEqual(str(invalid_vec)+" is not in space",str(context.exception))
    def test_vec2code_unsafe(self):
        invalid_vec=[1,1,0,0]
        self.assertEqual(None,vec2code(invalid_vec))
    def test_vec2codes(self):
        codes=['A','T','C','G']
        vectors=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        self.assertEqual(codes,vec2codes(vectors))
    def test_codes2vec(self):
        codes=['A','T','C','G']
        vectors=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        self.assertEqual(vectors,codes2vec(codes))
    def test_vec2codes_safe(self):
        invalid_vec=[1,0,0,1]
        vectors=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,0,0,0],invalid_vec]
        with self.assertRaises(Exception) as context:
            vec2codes(vectors)
        self.assertEqual(str(invalid_vec)+" is not in space",str(context.exception))
    def test_codes2vec_safe(self):
        invalid_code='I'
        codes=['A','T','C','G',invalid_code]
        with self.assertRaises(Exception) as context:
            codes2vec(codes)
        self.assertEqual(str(invalid_code)+" is not in space",str(context.exception))
    def test_vec2codes_unsafe(self):
        invalid_vec=[1,0,0,1]
        vectors=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,0,0,0],invalid_vec]
        self.assertEqual(None,vec2codes(vectors))
    def test_codes2vec_unsafe(self):
        invalid_code='I'
        codes=['A','T','C','G',invalid_code]
        self.assertEqual(None,codes2vec(codes))
if __name__=="__main__":
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestDNAVector)
    unittest.main()

