import json,tempfile,unittest
from pathlib import Path
import numpy as np
from src.a32.budget import Book,BudgetStop,totals,append_jsonl

class Budgets(unittest.TestCase):
    def fixture(self,stage,gate='PASS'):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);p=Path(tmp.name)
        cfg={'budgets':{'cpu_charged_seconds':1800,'material_states_total':64,'states_2d':56,'states_3d':8,'complex_RHS':8192}}
        (p/'a32_config.json').write_text(json.dumps(cfg));(p/'results/a32').mkdir(parents=True)
        (p/'results/a32/E0_GATE.json').write_text(json.dumps({'status':gate}))
        return p,Book(p,stage)

    def test_E0_rejects_native_state_even_if_gate_exists(self):
        p,b=self.fixture('e0')
        with self.assertRaises(BudgetStop):b.reserve_state(np.zeros(576))
        self.assertEqual(totals(p)['new_material_state_evaluations'],0)

    def test_failed_gate_rejects_state(self):
        p,b=self.fixture('shadow','FAIL')
        with self.assertRaises(BudgetStop):b.reserve_state(np.zeros(576))

    def test_all_resolutions_and_repeats_count(self):
        p,b=self.fixture('shadow')
        # Ledger arithmetic only: no backend or physical fixture is invoked.
        b.reserve_state(np.zeros(64),resolution=8,purpose='FAKE_LEDGER')
        b.reserve_state(np.zeros(576),purpose='FAKE_LEDGER')
        self.assertEqual(totals(p)['new_material_state_evaluations'],2)

    def test_shadow_cap_is_not_reset_by_new_Book(self):
        p,b=self.fixture('shadow');b.reserve_state(count=24,purpose='FAKE_LEDGER')
        c=Book(p,'shadow')
        with self.assertRaises(BudgetStop):c.reserve_state(purpose='FAKE_LEDGER')

    def test_rhs_batch_counts_columns(self):
        p,b=self.fixture('shadow');b.reserve_rhs(8,'FAKE_JVP');b.reserve_rhs(8,'FAKE_VJP')
        self.assertEqual(totals(p)['new_LU_RHS'],16)

    def test_3D_reserve_cannot_be_spent_on_2D(self):
        p,b=self.fixture('vector-check');b.reserve_state(count=8,physical_dimension=3,purpose='FAKE_LEDGER')
        with self.assertRaises(BudgetStop):b.reserve_state(physical_dimension=3,purpose='FAKE_LEDGER')

    def test_later_job_receipt_updates_live_CPU_limit(self):
        p,b=self.fixture('shadow')
        append_jsonl(p/'results/a32/COST_LEDGER.jsonl',{'charged_seconds':1800})
        with self.assertRaises(BudgetStop):b.check()

if __name__=='__main__':unittest.main()
