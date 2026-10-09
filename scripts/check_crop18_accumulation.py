"""Numerically compare the actual training accumulation helper with full-batch SGD."""
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from train import accumulation_step


def check(accumulation, microsteps):
    torch.manual_seed(123)
    model = torch.nn.Linear(3, 2).double()
    reference = copy.deepcopy(model)
    x = torch.randn(microsteps * 2, 3, dtype=torch.float64)
    y = torch.randn(microsteps * 2, 2, dtype=torch.float64)
    optimizers = [torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
                  for m in (model, reference)]
    schedulers = [torch.optim.lr_scheduler.StepLR(o, step_size=1, gamma=0.8)
                  for o in optimizers]
    updates = 0
    for i in range(microsteps):
        loss = torch.nn.functional.mse_loss(model(x[2*i:2*i+2]), y[2*i:2*i+2])
        updated = accumulation_step(loss, optimizers[0], schedulers[0], i, microsteps, accumulation)
        if updated:
            updates += 1
            start = (i // accumulation) * accumulation * 2
            end = (i + 1) * 2
            optimizers[1].zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(reference(x[start:end]), y[start:end]).backward()
            optimizers[1].step()
            schedulers[1].step()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            assert schedulers[0].state_dict() == schedulers[1].state_dict()
    assert updates == (microsteps + accumulation - 1) // accumulation


for settings in [(1, 5), (2, 6), (2, 5), (4, 3)]:
    check(*settings)
print('PASS: default path, full groups, partial final groups, per-update LR')
