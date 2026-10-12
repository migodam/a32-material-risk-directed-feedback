"""Eligibility checks use synthetic rows, never a physical state or target."""
import json, unittest
from pathlib import Path
from src.a32.statistics import e0_statistics, parent_family
from src.a32.e0 import gate_from_statistics


class GateCoverage(unittest.TestCase):
    def fixture(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/'a32_config.json').read_text())
        rows=[]
        for parent in cfg['parent_ids']:
            for bg in cfg['backgrounds']:
                for method,error in [('N3',1.),(cfg['primary_method'],.8)]:
                    rows.append(dict(parent_id=parent,case_id=f'SYNTHETIC_{parent}_{bg}',
                                     background_id=bg,family=parent_family(parent),method=method,
                                     success=True,status='OPTIMAL',fixed_mass_squared_error=error,
                                     delta_nrmse=.5,support_iou=.5,edge_f1=.5))
        return cfg,rows

    def gate(self,cfg,rows):
        return gate_from_statistics(e0_statistics(rows,cfg,methods=['N3',cfg['primary_method']]),cfg)

    def test_complete_gain_and_authoritative_gate(self):
        cfg,rows=self.fixture();gate=self.gate(cfg,rows)
        self.assertEqual(gate['status'],'PASS')
        self.assertEqual(gate['primary_cases'],24)
        self.assertAlmostEqual(gate['relative_equal_parent_MSE_gain'],.2)

    def test_missing_background_holds(self):
        cfg,rows=self.fixture();rows.pop(1)
        self.assertEqual(self.gate(cfg,rows)['status'],'HOLD')

    def test_same_row_count_with_duplicate_background_holds(self):
        cfg,rows=self.fixture();rows[3]=dict(rows[1])
        self.assertEqual(self.gate(cfg,rows)['status'],'HOLD')

    def test_failed_primary_not_removed_from_pairs(self):
        cfg,rows=self.fixture();rows[1]['success']=False
        self.assertEqual(self.gate(cfg,rows)['status'],'HOLD')

    def test_undefined_MSE_holds(self):
        cfg,rows=self.fixture();rows[1]['fixed_mass_squared_error']=float('nan')
        self.assertEqual(self.gate(cfg,rows)['status'],'HOLD')

    def test_unexpected_parent_holds(self):
        cfg,rows=self.fixture();rows.append(dict(rows[1],parent_id=999999))
        self.assertEqual(self.gate(cfg,rows)['status'],'HOLD')

    def test_ancillary_metric_is_not_an_extra_gate(self):
        cfg,rows=self.fixture();rows[1]['edge_f1']=None
        self.assertEqual(self.gate(cfg,rows)['status'],'PASS')


if __name__=='__main__':unittest.main()
