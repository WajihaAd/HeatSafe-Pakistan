import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'streamlit_app'))
from rag_engine import documents, retrieve, answer_guidance
class RagTests(unittest.TestCase):
    def test_source_metadata(self):
        for doc in documents():
            for key in ('id','title','section','url','text'):self.assertTrue(doc.get(key))
    def test_supported_worker_question(self):
        r=answer_guidance('What rest shade water precautions should outdoor workers take?')
        self.assertIn('CDC-HEAT-1',r['sources'])
    def test_unsupported_question_abstains(self):
        r=answer_guidance('How to repair a satellite rocket propulsion nozzle?')
        self.assertEqual(r['method'],'abstained')
    def test_no_fabricated_citations(self):
        r=answer_guidance('How can workers avoid heat stress?')
        self.assertTrue(set(r['sources']).issubset({d['id'] for d in documents()}))
if __name__=='__main__':unittest.main()
