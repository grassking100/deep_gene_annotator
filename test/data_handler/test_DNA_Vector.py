import unittest
from . import SeqConverter,SeqException,CodeException
class TestDNAVector(unittest.TestCase):
    def test_code2vec(self):
        converter = SeqConverter()
        codes=['A','T','C','G','a','t','c','g']
        vectors=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        for index in range(len(codes)):
            self.assertEqual(vectors[index%4],converter.code2vec(codes[index]))
    def test_code2vec_exception(self):
        converter = SeqConverter()
        codes=['i']
        with self.assertRaises(CodeException) as context:
            converter.code2vec(codes[0])
    def test_vec2code(self):
        converter = SeqConverter()
        codes=['A','T','C','G']
        vectors=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        for index in range(len(vectors)):
            self.assertEqual(codes[index],converter.vec2code(vectors[index]))
    def test_vec2code_exception(self):
        converter = SeqConverter()
        invalid_vec=[1,1,0,0]
        with self.assertRaises(CodeException) as context:
            converter.vec2code(invalid_vec)
        #self.assertEqual(str(invalid_vec)+" is not in space",str(context.exception))
    def test_vecs2seq(self):
        codes=['A','T','C','G']
        vectors=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        converter = SeqConverter()
        self.assertEqual(codes,converter.vecs2seq(vectors,join=False))
    def test_seq2vecs(self):
        codes=['A','T','C','G']
        vectors=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        converter = SeqConverter()
        self.assertEqual(vectors,converter.seq2vecs(codes))
    def test_vecs2seq_exception(self):
        invalid_vec=[1,0,0,1]
        vectors=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,0,0,0],invalid_vec]
        converter = SeqConverter()
        with self.assertRaises(SeqException) as context:
            converter.vecs2seq(vectors)
        #self.assertEqual(str(invalid_vec)+" is not in space",str(context.exception))
    def test_seq2vecs_exception(self):
        invalid_code='I'
        converter = SeqConverter()
        codes=['A','T','C','G',invalid_code]
        with self.assertRaises(SeqException) as context:
            converter.seq2vecs(codes)
        #self.assertEqual(str(invalid_code)+" is not in space",str(context.exception))
if __name__=="__main__":
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestDNAVector)
    unittest.main()

