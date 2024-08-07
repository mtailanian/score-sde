import torch
import numpy as np
import matplotlib.pyplot as plt
from configs.ve import anomaly_256_ncsnpp_continuous as configs
import plotly.graph_objs as go
import plotly.express as px
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from scipy import interpolate
import plotly.io as pio


def get_ddpm_params(config, num_diffusion_timesteps=1000):
    # parameters need to be adapted if number of time steps differs from 1000
    beta_start = config.model.beta_min / config.model.num_scales
    beta_end = config.model.beta_max / config.model.num_scales
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
        'beta_min': beta_start * (num_diffusion_timesteps - 1),
        'beta_max': beta_end * (num_diffusion_timesteps - 1),
        'num_diffusion_timesteps': num_diffusion_timesteps
    }


def generate_trajectory(starting_point, ts_lin, ts, params):
    # trajectory
    xi = [starting_point]
    for t in ts:
        epsilon = torch.randn(1)
        i = torch.argmin((ts_lin - t).abs())
        xi_t = np.sqrt(params['alphas'][i]) * xi[-1] + np.sqrt(1 - params['alphas'][i]) * epsilon
        xi.append(xi_t.item())
    xi = np.array(xi)
    xi = 1000 * (xi + 3) / 6
    return xi


def generate_ode_trajectory(starting_point, ts_lin, ts, params):
    xi = [starting_point]
    for t in ts:
        i = torch.argmin((ts_lin - t).abs())
        xi_t = np.sqrt(params['alphas'][i]) * xi[-1] + 0.003 * t * starting_point
        xi.append(xi_t.item())
    xi = np.array(xi)
    xi = 1000 * (xi + 3) / 6
    return xi


def generate_ode_chart():

    T = 2500
    config = configs.get_config()
    params = get_ddpm_params(config, T)

    x0 = torch.cat([
        torch.randn(100000) * 0.4 + 0.1,
        torch.randn(100000) * 0.6 + 2,
        torch.randn(100000) * 0.5 - 1.5,
        ])
    ts_lin = torch.from_numpy(np.linspace(0, 1, T))
    ts = ts_lin ** 3
    x = [x0]
    # epsilon = torch.randn_like(x0)
    for t in ts:
        epsilon = torch.randn_like(x0)
        i = torch.argmin((ts_lin - t).abs())
        x.append(np.sqrt(params['alphas'][i]) * x[-1] + np.sqrt(1 - params['alphas'][i]) * epsilon)
        # x.append(params['sqrt_alphas_cumprod'][i] * x0 + params['sqrt_1m_alphas_cumprod'][i] * epsilon)
    x = torch.stack(x[1:]).T

    # bins = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 1000)
    bins = np.linspace(-3, 3, 1000)
    density = []
    for i in range(x.shape[1]):
        pdf, _ = np.histogram(x[:, i].numpy(), bins, density=False)
        density.append(pdf)
    density = np.stack(density).T
    density = gaussian_filter(density, sigma=25)

    x1 = generate_trajectory(-1.7, ts_lin, ts, params)
    x2 = generate_trajectory(-1.4, ts_lin, ts, params)
    x3 = generate_trajectory(-0.1, ts_lin, ts, params)
    x4 = generate_trajectory(0.2, ts_lin, ts, params)
    x5 = generate_trajectory(1.8, ts_lin, ts, params)
    x6 = generate_trajectory(2.2, ts_lin, ts, params)

    x1_ode = generate_ode_trajectory(-2.3, ts_lin, ts, params)
    x2_ode = generate_ode_trajectory(-1.2, ts_lin, ts, params)
    x3_ode = generate_ode_trajectory(0.2, ts_lin, ts, params)
    x4_ode = generate_ode_trajectory(1.3, ts_lin, ts, params)
    x5_ode = generate_ode_trajectory(2.3, ts_lin, ts, params)

    plt.imshow(density)
    for xx in [x1_ode, x2_ode, x3_ode, x4_ode, x5_ode]:
        plt.plot(xx, color='white')
    for xx in [x1, x2, x3, x4, x5, x6]:
        plt.plot(xx)
    plt.show()

    # PLOT
    colormap = plt.get_cmap('viridis')
    colors = [colormap(i / 6) for i in range(6)]
    colors = ['rgb({}, {}, {})'.format(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colors]

    fig = px.imshow(density, color_continuous_scale='cividis')  # 'magma'
    for xx in [x1_ode, x2_ode, x3_ode, x4_ode, x5_ode]:
        fig.add_trace(go.Scatter(x=np.arange(T + 1), y=xx, mode='lines', line=dict(color='azure')))
    for i, xx in enumerate([x1, x2, x3, x4, x5, x6]):
        fig.add_trace(go.Scatter(x=np.arange(T + 1), y=xx, mode='lines', line=dict(color=colors[i])))
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(0, 1000, 100)),
            ticktext=[f"{b:.2f}" for b in bins[::100]],
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=np.linspace(0, 2500, 20),
            ticktext=[f"{b:.1f}" for b in np.linspace(0, 1, 20)],
        ),
        font=dict(size=35),
        coloraxis_showscale=False,
        showlegend=False,
        xaxis_title='Time',
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
    )
    pio.write_image(fig, 'flow.png', width=1920, height=1080)
    fig.show()

    for i in [0, -1]:
        fig = go.Figure(data=go.Scatter(
            x=bins[:-1],
            y=density[:, i],
            mode='markers',
            marker=dict(
                size=10,
                color=density[:, i],  # Color by values
                colorscale='cividis',  # Choose a colorscale
                showscale=True  # Show color scale
            )
        ))
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
            paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper background
        )
        pio.write_image(fig, f'dist_{i}.png', width=1080, height=500)
        fig.show()

    print


def main():
    generate_ode_chart()


if __name__ == '__main__':
    main()
