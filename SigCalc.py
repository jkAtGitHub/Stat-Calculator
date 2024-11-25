import streamlit as st
import pandas as pd
from scipy.stats import t, binomtest, norm
import numpy as np
import base64
import math

st.set_page_config(layout="wide")

if 'clicked' not in st.session_state:
    st.session_state.clicked = False
if 'powercalc' not in st.session_state: 
    st.session_state.powercalc = False
if 'lift' not in st.session_state:
    st.session_state.lift = 0

def calc_sample_size_mean(variance, mde, alpha, beta):
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(1 - beta)
    return round(((z_alpha + z_beta)**2 * 2 * variance) / mde**2)

# Calculate sample size for proportion-based metrics
def calc_sample_size_proportion(p, mde, alpha, beta):
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(1 - beta)
    variance = p * (1 - p)
    return round(((z_alpha + z_beta)**2 * variance) / mde**2)


def cal_z_val_for_significance(sides, sig_threshold):
    confidence = 1 - sig_threshold
    if sides == 1:
        percentile = confidence
    else:
        percentile = confidence + sig_threshold / 2

    z = norm(loc=0, scale=1).ppf(percentile)
    return z

def calc_prop_diff(size_treatment, size_control, prop_treatment, prop_control, 
                    metric_precision=2, expected_split_prop = 0.5, sides=2, sig_threshold=0.1, result_title="Results"):

    # Input validation
    if not all(0 <= p <= 1 for p in [prop_treatment, prop_control]):
        raise ValueError("Proportions must be between 0 and 1")
    if not all(isinstance(s, int) and s > 0 for s in [size_treatment, size_control]):
        raise ValueError("Sample sizes must be positive integers")

    z = cal_z_val_for_significance(sides, sig_threshold)

    var = prop_treatment * (1 - prop_treatment) / size_treatment + \
          prop_control * (1 - prop_control) / size_control
    se = np.sqrt(var)

    prop_diff = prop_treatment - prop_control
    metric_lift = prop_diff / prop_control
    
    lift_symbol = "%"
    lift_metric_precision = metric_precision
    if abs(metric_lift) < 0.01:
        metric_lift_num = metric_lift * 10000
        lift_symbol = "bps"
        lift_metric_precision = metric_precision - 2
    else:
        metric_lift_num = metric_lift * 100
    st.session_state.lift = metric_lift_num

    confint = prop_diff + np.array([-1, 1]) * z * se
    treatment_val_range = prop_control + confint

    # Use binomtest instead of binom_test
    HEALTH_TEST_pvalue = binomtest(size_control, size_control + size_treatment, expected_split_prop).pvalue
    SRM = HEALTH_TEST_pvalue < 0.05

    # Calculate the p-value for the hypothesis test.
    test_stat = prop_diff / se
    pvalue = sides * norm.cdf(-np.abs(test_stat))

    # Calculate the significance of the results.
    significant = "Significant" if pvalue < sig_threshold else "Not Significant"

    # Create a dictionary containing the results of the hypothesis test.
    d = {
        "Treatment proportion": f"{round(prop_treatment * 100, metric_precision)}%",
        "Control proportion": f"{round(prop_control * 100, metric_precision)}%",
        "Lift" : f"{round(metric_lift_num, lift_metric_precision)}{lift_symbol}",
        "p-value": round(pvalue, 5),
        "Significant": significant,
        "Confidence Interval (Difference)": f"{round(confint[0] * 100, metric_precision)}% to {round(confint[1] * 100, metric_precision)}%",
        "Confidence Interval (Treatment Value)": f"{round(treatment_val_range[0] * 100, metric_precision)}% to {round(treatment_val_range[1] * 100, metric_precision)}%",
        "Users in Control": f"{size_control:,}",
        "Users in Treatment": f"{size_treatment:,}",
        "Sample Ratio Mismatch detected?" : SRM
    }

    return pd.DataFrame([d], index=[result_title])

def calc_mean_diff(size_treatment, size_control, mean_treatment, mean_control, 
                    var_treatment, var_control, metric_precision=2, sides=2, 
                    sig_threshold=0.1, expected_split_prop = 0.5, unit_prefix="", unit_suffix="", result_title="Results"):

    # Input validation
    if not all(isinstance(s, int) and s > 0 for s in [size_treatment, size_control]):
        raise ValueError("Sample sizes must be positive integers")
    if not all(v >= 0 for v in [var_treatment, var_control]):
        raise ValueError("Variances must be non-negative")

    z = cal_z_val_for_significance(sides, sig_threshold)

    var = var_treatment / size_treatment + var_control / size_control
    se = np.sqrt(var)

    mean_diff = mean_treatment - mean_control
    metric_lift = mean_diff / mean_control
    
    lift_symbol = "%"
    lift_metric_precision = metric_precision
    if abs(metric_lift) < 0.01:
        metric_lift_num = metric_lift * 10000
        lift_symbol = "bps"
        lift_metric_precision = metric_precision - 2
    else:
        metric_lift_num = metric_lift * 100

    st.session_state.lift = metric_lift_num

    confint = mean_diff + np.array([-1, 1]) * z * se
    treatment_val_range = mean_control + confint
    test_stat = mean_diff / se

    # Dynamic test selection (Z vs. t)
    if size_treatment < 30 or size_control < 30:
        
        pvalue = sides * t.cdf(-np.abs(test_stat), df=min(size_treatment, size_control) - 1)
    else:
        pvalue = sides * norm.cdf(-np.abs(test_stat))

    significant = "Significant" if pvalue < sig_threshold else "Not Significant"
    
    HEALTH_TEST_pvalue = binomtest(size_control, size_control + size_treatment, expected_split_prop).pvalue
    SRM = HEALTH_TEST_pvalue < 0.05

    # Calculate Cohen's d
    cohen_d = mean_diff / np.sqrt((var_treatment + var_control) / 2)

    d = {
          "Treatment Mean" : f"{unit_prefix}{round(mean_treatment, metric_precision)}{unit_suffix}",  
          "Control Mean" : f"{unit_prefix}{round(mean_control, metric_precision)}{unit_suffix}",
          "Lift" : f"{round(metric_lift_num, lift_metric_precision)}{lift_symbol}",
          "p-value" : round(pvalue, metric_precision),
          "Significant" : significant,
          "Confidence Interval (Difference)" : f"{unit_prefix}{round(confint[0], metric_precision)} to {unit_prefix}{round(confint[1] , metric_precision)}{unit_suffix}",
          "Confidence Interval (Treatment Value)" : f"{unit_prefix}{round(treatment_val_range[0], metric_precision)}{unit_suffix} to {unit_prefix}{round(treatment_val_range[1] , metric_precision)}{unit_suffix}",
          "Users in Control" : f"{size_control:,}" ,
          "Users in Treatment" : f"{size_treatment:,}" ,
          "Sample Ratio Mismatch detected?" : SRM,
          "Cohen's d": round(cohen_d, metric_precision)
      }

    return pd.DataFrame([d], index=[result_title])

def click_button(param):
    if param == "Sig Calc":
        st.session_state.clicked = True
    elif param == "Power Calc":
        st.session_state.powercalc = True

def page_significance_calculator():
 
    # Streamlit app
    st.title("Significance Calculator")

    def color_significant_val(val):
        if val < sig_threshold: 
            if st.session_state.lift > 0:
                color = 'green'
            else: 
                color = 'red'
        else: 
            color = 'gray'
        return f'background-color: {color}'

    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        test_type = st.selectbox("Select metric type", ["Proportion", "Mean"])
    with c2: 
        precision = st.slider("Precion for results", 0, 4, 1, 1)
    with c3:
        sig_threshold = st.number_input("Significance Threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    with c4:
        cProp = st.number_input('Designed Control Proportion (%)',min_value=0.0, max_value=100.0, value= 50.0, step = 0.1)  
    st.divider()
    if test_type:
        st.subheader(f"Test for differences in {test_type}")
    col1, col2 = st.columns(2)
    
    with col1:
        if test_type == "Proportion":
            c1, c2 = st.columns(2)
            with c1: 
                size_treatment = st.number_input("Treatment Sample Size", min_value=1, value=10000, step=1000)
                prop_treatment = st.number_input("Treatment Proportion", min_value=0.0000, max_value=1.0000, value=.2100, step=0.0010)
            with c2:
                size_control = st.number_input("Control Sample Size", min_value=1, value=10000, step=1000)
                prop_control = st.number_input("Control Proportion", min_value=0.0000, max_value=1.0000, value=0.2000, step=0.0010)
                result_title = st.text_input("Results Title", value="Results")

        elif test_type == "Mean":
            c1, c2 = st.columns(2)
            with c1: 
                size_treatment = st.number_input("Treatment Sample Size", min_value=1, value=10000, step=1000)
                var_treatment = st.number_input("Treatment Variance", min_value=0.000, value=1.000, step=0.1)
                mean_treatment = st.number_input("Treatment Mean", value=5.000, step=0.100)
            with c2:
                size_control = st.number_input("Control Sample Size", min_value=1, value=10000, step=1000)
                mean_control = st.number_input("Control Mean", value=4.500, step=0.100)
                var_control = st.number_input("Control Variance", min_value=0.000, value=1.000, step=0.1000)
                result_title = st.text_input("Results Title", value="Results")

        st.button('Calculate', on_click=click_button, args=["Sig Calc"])

    with col2:
        if st.session_state.clicked:
            if test_type == "Proportion":
                try:
                    
                    EXPECTED_PROP = cProp/100
                    result = calc_prop_diff(size_treatment, size_control, prop_treatment, prop_control, 
                                            metric_precision=precision, 
                                            expected_split_prop=EXPECTED_PROP, 
                                            sig_threshold= sig_threshold, result_title=result_title)
                    st.dataframe(result.T, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            elif test_type == "Mean":
                try:
                    EXPECTED_PROP = cProp/100
                    result = calc_mean_diff(size_treatment, size_control, mean_treatment, mean_control,
                                    var_treatment, var_control, metric_precision=precision, 
                                    expected_split_prop=EXPECTED_PROP, 
                                    sig_threshold= sig_threshold, result_title=result_title)
                    st.dataframe(result.T, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")

# Power Analysis Page
def page_power_analysis():
    st.title("Power Analysis")
    st.write("Calculate sample size and weeks required for a mean-based or proportion-based metric.")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: 
        metric_type = st.selectbox("Select metric type", ["Proportion", "Mean"])
    with c2: 
        precision = st.slider("Precion for results", 0, 4, 1, 1)
    with c3:
        alpha = st.number_input("Significance Level (α)", min_value=0.01, max_value=0.2, value=0.01, step=0.01)
    with c4:
        #cProp = st.number_input('Designed Control Proportion (%)',min_value=0.0, max_value=100.0, value= 50.0, step = 0.1)  
        calc_type = st.selectbox("Choose MDE Type", ["Absolute", "Relative"])
    with c5: 
        beta = 1 - st.slider("Power (1 - β)", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
  
    st.divider()
    if metric_type:
        st.subheader(f"Power analysis for differences in {metric_type}")

    col1, col2 = st.columns(2)
    with col1: 
        # Common inputs
        # alpha = st.slider("Significance Level (α)", min_value=0.01, max_value=0.2, value=0.05, step=0.01)
        # beta = 1 - st.slider("Power (1 - β)", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
        weekly_traffic = st.number_input("Weekly Traffic", min_value=1, value=1000)
        lower_mde = st.number_input(f"Lower Bound MDE ({calc_type} lift)", min_value=0.001, value=0.01, step=0.01)
        upper_mde = st.number_input(f"Upper Bound MDE ({calc_type} lift)", min_value=0.001, value=0.11, step=0.01)
        step_mde = st.number_input(f"Step Value for MDE ({calc_type} lift)", min_value=0.001, value=0.01, step=0.01)

        # Inputs for mean-based or proportion-based
        if metric_type == "Mean":
            if calc_type == "Relative":
                mean = st.number_input("Baseline (Control) mean value", min_value=0, value=100)
            variance = st.number_input("Variance of the Metric", min_value=0.01, value=10.0, step=0.01)
        else:
            proportion = st.number_input("Base Proportion (e.g., 0.5)", min_value=0.01, max_value=1.0, value=0.5)

        st.button('Calculate', key = "Calculate Power", on_click=click_button, args=["Power Calc"])

        # if st.button("Calculate", key = "Calculate Power", on_click = click_button, args=["Power Calc"]):
           
        with col2: 
            if st.session_state.powercalc: 
                 # Generate MDE range
                mde_values = np.arange(lower_mde, upper_mde + step_mde, step_mde)
                sample_sizes = []
                weeks_required = []
                mde_values_str = []

                # Calculate for each MDE
                for mde in mde_values:
                    mde_values_str.append(f"{round(mde * 100, precision)}%")

                    if metric_type == "Mean":
                        if calc_type == "Relative":
                            mde = mde * mean
                        sample_size = calc_sample_size_mean(variance, mde, alpha, beta)
                    else:
                        if calc_type == "Relative":
                            mde = mde * proportion
                        sample_size = calc_sample_size_proportion(proportion, mde, alpha, beta)
                    
                    sample_sizes.append(sample_size)
                    actual_weeks = sample_size//weekly_traffic
                    remaining_days = math.ceil((sample_size%weekly_traffic)*7/weekly_traffic)
                    if remaining_days == 7: 
                        actual_weeks += 1
                        remaining_days = 0

                    weeks_required.append(f"{actual_weeks} Wk {remaining_days} D")
                
                # Display results
                results = {
                    f"MDE({calc_type}%)" : mde_values_str, 
                    f"MDE ({calc_type})": mde_values,
                    "Sample Size": sample_sizes,
                    "Weeks Required": weeks_required
                }
                results_df = pd.DataFrame(results)
                # Display table
                st.subheader("Results")
                st.dataframe(results_df[[f"MDE({calc_type}%)", "Sample Size", "Weeks Required"]], use_container_width=True)

                # Plot results
                st.subheader("Visualization")
                st.line_chart(results_df, x=f"MDE ({calc_type})", y = "Sample Size", use_container_width=True)
# Multi-page App Navigation
def main():
    st.sidebar.title("Navigation")
    pages = {
        "Power Analysis": page_power_analysis,
        "Significance Calculator": page_significance_calculator
    }
    choice = st.sidebar.radio("Go to", list(pages.keys()))
    st.sidebar.divider()
    # st.sidebar.write("Developed by")
    # st.sidebar.subheader("JK Vijayaraghavan")
    st.sidebar.markdown(
        """Developed by <br><b>JK Vijayaraghavan</b>&nbsp;<a href="https://www.linkedin.com/in/simplyjk/">
        <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width="25">
        </a>""",
        unsafe_allow_html=True,
        )
    st.sidebar.divider()
    st.sidebar.write("Provide Feedback")
    sentiment_mapping = ["one", "two", "three", "four", "five"]
    selected = st.sidebar.feedback("stars")
    if selected is not None:
        if selected >= 3: 
            st.sidebar.markdown(f"{sentiment_mapping[selected].title()} stars! That means a lot :smiley:!")
        elif selected > 1: 
            st.sidebar.markdown(f"{sentiment_mapping[selected].title()} stars... I'll work on that :pensive:")
        else: 
            st.sidebar.markdown(f"{sentiment_mapping[selected].title()} star... I'll work on that :pensive:")

    pages[choice]()

if __name__ == "__main__":
    main()