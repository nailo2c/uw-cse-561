import unittest

import torch

from homeworks.multinomial_log_regression import J_loss, L_loss, accuracy


class TestMultinomialLogRegression(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor(
            [[2.0, 1.0, 0.0], [0.5, 1.5, 2.5], [1.0, 3.0, 2.0]]
        )
        self.labels = torch.tensor([0, 2, 1])

    def test_L_loss_returns_scalar_tensor(self):
        loss = L_loss(self.logits, self.labels)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_J_loss_returns_scalar_tensor(self):
        loss = J_loss(self.logits, self.labels)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_accuracy_returns_fraction_correct(self):
        self.assertAlmostEqual(accuracy(self.logits, self.labels), 1.0)


if __name__ == "__main__":
    unittest.main()
