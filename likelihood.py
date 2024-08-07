# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""

import torch
import numpy as np
from scipy import integrate
from models import utils as mutils


def get_div_fn_tai(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    x.requires_grad_(False)
    # return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))
    return grad_fn_eps * eps

  return div_fn

def get_div_fn_tai_sum_chan(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    x.requires_grad_(False)
    return torch.sum(grad_fn_eps * eps, dim=1, keepdim=True)

  return div_fn

def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    x.requires_grad_(False)
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

  return div_fn


def get_likelihood_fn(sde, inverse_scaler, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
  """Create a function to compute the unbiased log-likelihood estimate of a given data point.

  Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

  Returns:
    A function that a batch of data points and returns the log-likelihoods in bits/dim,
      the latent code, and the number of function evaluations cost by computation.
  """

  def drift_fn(model, x, t):
    """The drift function of the reverse-time SDE."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)
    # Probability flow ODE is a special case of Reverse SDE
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]  # f - g^2 * score

  def div_fn(model, x, t, noise):
    return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)

  def div_fn_tai(model, x, t, noise):
    return get_div_fn_tai(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)

  def div_fn_tai_sum_chan(model, x, t, noise):
    return get_div_fn_tai_sum_chan(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)

  def likelihood_fn(model, data):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      model: A score model.
      data: A PyTorch tensor.

    Returns:
      bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
      z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    with torch.no_grad():
      shape = data.shape

      #
      if hutchinson_type == 'Gaussian':
        epsilon = torch.randn_like(data)
      elif hutchinson_type == 'Rademacher':
        epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
      else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
      #
      # n_iters = 10  # n_iters = 1 is equivalent to original Hutchinson's estimator
      # epsilon = torch.randint(size=(n_iters, *shape), low=0, high=2, device=data.device).float() * 2 - 1.
      #

      def ode_func(t, x):
        sample = mutils.from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
        vec_t = torch.ones(sample.shape[0], device=sample.device) * t
        drift = mutils.to_flattened_numpy(drift_fn(model, sample, vec_t))

        #
        logp_grad = mutils.to_flattened_numpy(div_fn(model, sample, vec_t, epsilon))
        #
        estimation = []
        # for iter in range(n_iters):
        #   # estimation.append(div_fn_tai(model, sample, vec_t, epsilon))
        #   # estimation.append(div_fn_tai(model, sample, vec_t, epsilon[iter]))
        #   estimation.append(div_fn_tai(model, sample, vec_t, torch.randint_like(data, low=0, high=2).float() * 2 - 1.))
        # estimation = torch.stack(estimation).mean(dim=0)
        # logp_grad = mutils.to_flattened_numpy(torch.sum(estimation, dim=tuple(range(1, len(x.shape)))))
        #

        return np.concatenate([drift, logp_grad], axis=0)

      init = np.concatenate([mutils.to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
      solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      zp = solution.y[:, -1]
      z = mutils.from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
      delta_logp = mutils.from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
      prior_logp = sde.prior_logp(z)
      bpd = -(prior_logp + delta_logp) / np.log(2)
      N = np.prod(shape[1:])
      bpd = bpd / N
      # A hack to convert log-likelihoods to bits/dim
      offset = 7. - inverse_scaler(-1.)
      bpd = bpd + offset
      return bpd, z, nfe

      # For VE
      prior_logp = -1 / 2. * np.log(2 * np.pi * sde.sigma_max ** 2) - z ** 2 / (2 * sde.sigma_max ** 2)

      vec_t = torch.ones(data.shape[0], device=data.device) * sde.T
      delta_logp = get_div_fn_tai(lambda xx, tt: drift_fn(model, xx, tt))(z, vec_t, epsilon)

      logp = prior_logp + delta_logp


      #


      n_hutchinson_samples = 5
      logp = []
      for _ in range(n_hutchinson_samples):

        deq_data = 255 * inverse_scaler(data)
        deq_data = (deq_data + torch.rand_like(deq_data)) / 256.0
        # deq_data = scaler(deq_data)  # scaler and inverse_scaler are actually x=y

        init = np.concatenate([mutils.to_flattened_numpy(deq_data), np.zeros((shape[0],))], axis=0)
        solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
        zp = solution.y[:, -1]
        z = mutils.from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)

        epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
        prior_logp = -1 / 2. * np.log(2 * np.pi * sde.sigma_max ** 2) - z ** 2 / (2 * sde.sigma_max ** 2)
        delta_logp = get_div_fn_tai(lambda xx, tt: drift_fn(model, xx, tt))(z, vec_t, epsilon)

        logp.append(prior_logp + delta_logp)
      logp = torch.stack(logp).mean(dim=0)

      #
      #
      #
      t = solution.t
      num_time_steps = len(t)  # = solution.y.shape[1]
      all_trajectories = mutils.from_flattened_numpy(solution.y[:-shape[0], :], [*shape, num_time_steps])

      # all_trajectories is of size [t, batch, channel, height, width]
      all_trajectories = all_trajectories.permute(4, 0, 1, 2, 3)
      all_trajectories = all_trajectories.type(torch.float32).to("cuda")

      f_grad = []
      for i, sample in enumerate(all_trajectories):
        vec_t = torch.ones(sample.shape[0], device=sample.device) * t[i]

        n_hutchinson_samples = 50
        epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
        # f_grad_t = div_fn_tai(model, sample, vec_t, epsilon)
        f_grad_t = div_fn_tai_sum_chan(model, sample, vec_t, epsilon)
        for _ in range(n_hutchinson_samples - 1):
          epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
          # f_grad_t += div_fn_tai(model, sample, vec_t, epsilon)
          f_grad_t += div_fn_tai_sum_chan(model, sample, vec_t, epsilon)
        f_grad_t /= n_hutchinson_samples
        f_grad.append(f_grad_t)
      f_grad = torch.stack(f_grad)

      f_grad_integral = torch.trapz(f_grad, torch.from_numpy(t).to(f_grad.device), dim=0)
      f_grad_integral_sum = torch.sum(torch.from_numpy(np.diff(np.insert(t, 0, 0))[:, None, None, None, None]).to(f_grad.device) * f_grad, dim=0)

      from utils import imshow_tensor
      imshow_tensor(f_grad_integral)

      zp = solution.y[:, -1]
      z = mutils.from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)

      tai_prior_logp = ve_prior_logp_no_reduce(sde, z)
      tai_prior_logp = ve_prior_logp_no_reduce_sum_chan(sde, z)

      log_p0 = tai_prior_logp + f_grad_integral

      bpd = -(tai_prior_logp + f_grad_integral) / np.log(2)
      N = np.prod(shape[1:])
      bpd = bpd / N
      # A hack to convert log-likelihoods to bits/dim
      offset = 7. - inverse_scaler(-1.)
      bpd = bpd + offset


      # prior_logp = prior_logp_no_reduce(sde, z / 256)
      # prior_logp = subvp_prior_logp_no_reduce(z)
      # likelihood = torch.exp(-torch.mean(z ** 2, dim=1, keepdim=True) * 0.5)

      def ve_prior_logp_no_reduce(self, z):
        # N = np.prod(z.shape[1:])
        N = 1
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - z ** 2 / (2 * self.sigma_max ** 2)

      def ve_prior_logp_no_reduce_sum_chan(self, z):
        N = z.shape[1]
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=1, keepdim=True) / (2 * self.sigma_max ** 2)

      def subvp_prior_logp_no_reduce(z):
        shape = z.shape
        N = np.prod(shape[1:])
        # return -N / 2. * np.log(2 * np.pi) - z ** 2 / 2.
        return -1 / 2. * np.log(2 * np.pi) - z ** 2 / 2.

      #
      #
      #
      #
      # PLOT TRAJECTORIES
      # from utils import get_path, imshow_tensor
      # from PIL import Image
      # import random
      # import matplotlib.pyplot as plt
      #
      # n_trajectories = 50
      # channel = 0
      # image_idx = 0
      #
      # # label_path = get_path('data') / "mvtec" / "carpet" / "ground_truth" / "color" / "000_mask.png"
      # label_path = get_path('data') / "mvtec" / "grid" / "ground_truth" / "bent" / "000_mask.png"
      # # label_path = get_path('data') / "mvtec" / "hazelnut" / "ground_truth" / "crack" / "000_mask.png"
      #
      # label = torch.from_numpy(np.array(Image.open(label_path).convert('L').resize((256, 256)))).unsqueeze(0).float() / 255
      #
      # indexes_normal = (label == 0).nonzero(as_tuple=False)
      # indexes_anomaly = (label == 1).nonzero(as_tuple=False)
      #
      # chosen_normal = indexes_normal[torch.tensor(random.sample(range(len(indexes_normal)), n_trajectories))]
      # chosen_anomaly = indexes_anomaly[torch.tensor(random.sample(range(len(indexes_anomaly)), n_trajectories))]
      #
      # num_time_steps = solution.y.shape[1]
      # all_trajectories = mutils.from_flattened_numpy(solution.y[:-shape[0], :], [*shape, num_time_steps])
      #
      # normal_trajectories = all_trajectories[image_idx, channel, chosen_normal[:, 1], chosen_normal[:, 2], :]
      # anomalous_trajectories = all_trajectories[image_idx, channel, chosen_anomaly[:, 1], chosen_anomaly[:, 2], :]
      #
      # means = torch.cat([normal_trajectories, anomalous_trajectories]).mean(dim=0)
      # stds = torch.cat([normal_trajectories, anomalous_trajectories]).std(dim=0)
      # normal_trajectories = (normal_trajectories - means) / stds
      # anomalous_trajectories = (anomalous_trajectories - means) / stds
      #
      # plt.plot(normal_trajectories.T, 'g', alpha=0.2)
      # plt.plot(anomalous_trajectories.T, 'r', alpha=0.2)
      # plt.show()
      #
      #
      #
      #

      # # Build ODE trajectory for a pixel (i, j, channel)
      # i, j = 10, 10  # pixel indexes
      # for _ in range(20):
      #   i, j = np.random.randint(low=0, high=255), np.random.randint(low=0, high=255)
      #
      #   trajectory = []
      #   for t in range(solution.y.shape[1]):
      #     a = solution.y[:, t]
      #     a = mutils.from_flattened_numpy(a[:-shape[0]], shape).to(data.device).type(torch.float32)
      #     trajectory.append(a[0, channel, i, j].item())
      #   plt.plot(trajectory)
      # plt.show()

  return likelihood_fn

#
#
#
#
#
# %%


def tai_drift_fn(model, sde, x, t):
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)
  rsde = sde.reverse(score_fn, probability_flow=True)
  return rsde.sde(x, t)[0]  # f - g^2 * score


def tai_div_fn(model, sde, x, t, eps):
  with torch.enable_grad():
    x.requires_grad_(True)
    fn_eps = torch.sum(tai_drift_fn(model, sde, x, t) * eps)
    grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
  x.requires_grad_(False)
  # return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))
  return grad_fn_eps * eps


def tai_compute_z(data, model, sde):
  shape = data.shape

  # epsilon = torch.randn_like(data)
  epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.

  def tai_ode_fun(t, x):
    sample = mutils.from_flattened_numpy(x[:x.shape[0] // 2], shape).to(data.device).type(torch.float32)
    vec_t = torch.ones(sample.shape[0], device=sample.device) * t
    drift = mutils.to_flattened_numpy(tai_drift_fn(model, sde, sample, vec_t))
    logp_grad = mutils.to_flattened_numpy(tai_div_fn(model, sde, sample, vec_t, epsilon))
    return np.concatenate([drift, logp_grad], axis=0)

  def ve_prior_logp_no_reduce(self, z):
    N = 1
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - z ** 2 / (2 * self.sigma_max ** 2)

  vec_data = mutils.to_flattened_numpy(data)
  init = np.concatenate([vec_data, np.zeros_like(vec_data)], axis=0)
  solution = integrate.solve_ivp(tai_ode_fun, (1e-5, sde.T), init, rtol=1e-5, atol=1e-5, method='RK45')
  zp = solution.y[:, -1]
  z = mutils.from_flattened_numpy(zp[:vec_data.shape[0]], shape).to(data.device).type(torch.float32)
  prior_logp = ve_prior_logp_no_reduce(sde, z)

  t = solution.t
  # delta_logp = mutils.from_flattened_numpy(zp[vec_data.shape[0]:], shape).to(data.device).type(torch.float32)
  delta_logp_t = mutils.from_flattened_numpy(solution.y[vec_data.shape[0]:, :], (*shape, len(t))).to(data.device).type(torch.float32)
  delta_logp = torch.trapz(delta_logp_t, torch.from_numpy(t).to(delta_logp_t.device), dim=-1)

  px = prior_logp + delta_logp

  # results = []
  # for ti, x in zip(solution.t, solution.y.T):
  #   results.append(tai_ode_fun(ti, x))
  # results = np.stack(results)
  # delta_logp_t = results[:, vec_data.shape[0]:]
  # delta_logp = torch.trapz(torch.from_numpy(delta_logp_t).to(data.device), torch.from_numpy(t).to(data.device), dim=0)
  # delta_logp = mutils.from_flattened_numpy(delta_logp.cpu().numpy(), shape).to(data.device).type(torch.float32)
  return z
