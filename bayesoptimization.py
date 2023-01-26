import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] = 0.0

    return expected_improvement

kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
gpr = GaussianProcessRegressor(kernel=kernel,
                               alpha=1e-10,
                               optimizer="fmin_l_bfgs_b",
                               n_restarts_optimizer=0,
                               normalize_y=True,
                               copy_X_train=True,
                               random_state=0)

# def evaluation_function(x):
#     return np.sin(x)

if "df" not in st.session_state: 
    st.session_state.df = pd.DataFrame(columns=["x",
                                                "y_true"])

st.subheader("Bayesian Optimization Demo")
select = st.selectbox("Select a function", ["sin(x)", "x^2", "x^3", "x^4", "cos(x)", "exp(x)", "log(x)"])

if select == "sin(x)":
    evaluation_function = np.sin
elif select == "x^2":
    evaluation_function = lambda x: x**2
elif select == "x^3":
    evaluation_function = lambda x: x**3
elif select == "x^4":
    evaluation_function = lambda x: x**4
elif select == "cos(x)":
    evaluation_function = np.cos
elif select == "exp(x)":
    evaluation_function = np.exp
elif select == "log(x)":
    evaluation_function = np.log

select_optimum = st.selectbox("Select an optimum", ["min", "max"])

input_range = st.slider("Select a range", -20.0, 20.0, (0.0, 10.0))

x_points = np.linspace(input_range[0], input_range[1], 1000)
y = evaluation_function(x_points)


fig = go.Figure()
fig.add_trace(go.Scatter(x=x_points, y=y, mode='lines', name='True Function', line=dict(color='red')))
fig.add_trace(go.Scatter(x=st.session_state.df["x"], y=st.session_state.df["y_true"], mode='markers', name='True Function', marker=dict(color='red')))

if (len(st.session_state.df) > 0):
    gpr.fit(st.session_state.df["x"].values.reshape(-1, 1), st.session_state.df["y_true"].values)
    gpr.score(st.session_state.df["x"].values.reshape(-1, 1), st.session_state.df["y_true"].values)

    y_pred, y_std = gpr.predict(x_points.reshape(-1, 1), return_std=True)
    
    y_upper = y_pred + y_std
    y_lower = y_pred - y_std

    
    fig.add_trace(go.Scatter(x=x_points, y=y_pred, mode='lines', name='Predicted Function', line=dict(color='rgba(0,100,80,1)')))
    
    fig.add_trace(go.Scatter(x=np.concatenate([x_points, x_points[::-1]]),
                             y=np.concatenate([y_upper, y_lower[::-1]]),
                             fill='toself',
                             name = "Uncertainty",
                             fillcolor='rgba(0,100,80,0.2)',
                             line=dict(color='rgba(255,255,255,0)')))

    y_EI = expected_improvement(x_points.reshape(-1, 1), gpr, st.session_state.df["y_true"].values, greater_is_better=select_optimum=="max", n_params=1)

    
    y_EI = max(np.abs(y_pred))/max(np.abs(y_EI)) * y_EI
    
    fig.add_trace(go.Scatter(x=x_points, y=y_EI, mode='lines', name='Scaled Expected Improvement', line=dict(color='blue')))
      

st.text("Click on the graph twice to add a new point")
st.text("Refresh the page to reset the graph")

selected_points = plotly_events(fig, key="plotly_click")

# st.dataframe(st.session_state.df)

if len(selected_points) > 0:
    x_added = selected_points[0]['x']
    y_true = evaluation_function(x_added)
    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([{"x": x_added,
                                                                        "y_true": y_true,}])], ignore_index=True)
