"""Dataset counts and the frozen split."""
import json
import re
import unittest
from collections import Counter
from pathlib import Path
from chartbreak.data import load_viseval, make_split, apply_split

DATA = Path('data/visEval_dataset')


class DataCountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads((DATA / 'visEval.json').read_text())
        cls.cases = load_viseval(DATA)

    def test_counts(self):
        nq = lambda d: sum(len(r['nl_queries']) for r in d.values())
        single = json.loads((DATA / 'visEval_single.json').read_text())
        multiple = json.loads((DATA / 'visEval_multiple.json').read_text())
        self.assertEqual((len(self.raw), nq(self.raw)), (1150, 2524))
        self.assertEqual(len({r['db_id'] for r in self.raw.values()}), 146)
        self.assertEqual(len(json.loads((DATA / 'databases/db_tables.json').read_text())), 166)
        self.assertEqual((len(single), nq(single), len(multiple), nq(multiple)), (688, 1588, 462, 936))
        self.assertEqual(set(single) | set(multiple), set(self.raw))

    def test_feature_shares(self):
        multiple = json.loads((DATA / 'visEval_multiple.json').read_text())
        q = [(cid, r) for cid, r in self.raw.items() for _ in r['nl_queries']]
        share = lambda f: round(100 * sum(map(f, q)) / len(q), 1)
        self.assertEqual(share(lambda x: x[0] in multiple), 37.1)
        self.assertEqual(share(lambda x: x[1]['vis_obj']['sort'] is not None), 48.7)
        self.assertEqual(share(lambda x: bool(re.search(r'\bjoin\b', x[1]['vis_query']['VQL'], re.I))), 35.9)
        self.assertEqual(share(lambda x: bool(x[1]['vis_query']['data_part']['binning'])), 10.9)
        self.assertEqual(share(lambda x: bool(x[1]['vis_obj'].get('classify'))), 6.1)
        self.assertEqual(sum(c['features']['multi_table'] for c in self.cases), 936)

    def test_figure_1a_counts(self):
        counts = Counter((c['features']['chart_type'], c['features']['hardness']) for c in self.cases)
        levels = ['Easy', 'Medium', 'Hard', 'Extra Hard']
        expected = {'Bar': [269, 791, 399, 148], 'Pie': [242, 222, 78, 19], 'Line': [21, 64, 25, 13],
                    'Scatter': [51, 16, 5, 8], 'Stacked Bar': [2, 8, 22, 38],
                    'Grouping Scatter': [34, 23, 0, 1], 'Grouping Line': [3, 6, 6, 10]}
        for chart, row in expected.items():
            self.assertEqual([counts[(chart, h)] for h in levels], row, chart)

    def test_known_quirk(self):
        row = self.raw['2088@x_name@ASC']
        self.assertEqual((len(row['nl_queries']), len(row['query_meta'])), (1, 2))

    def test_frozen_split_follows_rule(self):
        split = json.loads(Path('splits/split_v1.json').read_text())
        seed, dev_dbs, eval_dbs = make_split(self.cases)
        self.assertEqual((split['seed'], split['dev_dbs'], split['eval_dbs']), (3, dev_dbs, eval_dbs))
        dev, ev = apply_split(self.cases, 'splits/split_v1.json')
        self.assertEqual((len(dev), len(ev)), (523, 2001))
        self.assertFalse({c['db_id'] for c in dev} & {c['db_id'] for c in ev})
        types = {c['features']['chart_type'] for c in self.cases}
        self.assertEqual({c['features']['chart_type'] for c in dev}, types)
        self.assertEqual({c['features']['chart_type'] for c in ev}, types)
        self.assertGreaterEqual(min(Counter(c['features']['chart_type'] for c in ev).values()), 14)


if __name__ == '__main__':
    unittest.main()
