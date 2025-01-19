# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


def plot_sales_trends(df):
    """
    Create basic visualizations of the sales data

    Parameters:
    df (pandas.DataFrame): DataFrame with columns: date, country, store, product, num_sold
    """
    # Set up the plotting style
    # plt.style.use('seaborn')

    # Create multiple plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Monthly sales trend
    monthly_sales = df.groupby(df["date"].dt.to_period("M"))["num_sold"].sum()
    monthly_sales.plot(ax=axes[0, 0], title="Monthly Sales Trend")
    axes[0, 0].set_xlabel("Month")
    axes[0, 0].set_ylabel("Total Sales")

    # 2. Sales by country
    country_sales = df.groupby("country")["num_sold"].sum().sort_values(ascending=True)
    country_sales.plot(kind="barh", ax=axes[0, 1], title="Sales by Country")
    axes[0, 1].set_xlabel("Total Sales")

    # 3. Product performance
    product_sales = df.groupby("product")["num_sold"].sum().sort_values(ascending=True)
    product_sales.plot(kind="barh", ax=axes[1, 0], title="Sales by Product")
    axes[1, 0].set_xlabel("Total Sales")

    # 4. Store comparison
    store_sales = df.groupby("store")["num_sold"].sum().sort_values(ascending=True)
    store_sales.plot(kind="barh", ax=axes[1, 1], title="Sales by Store")
    axes[1, 1].set_xlabel("Total Sales")

    plt.tight_layout()
    return fig


# %%

df = pd.read_csv("../data/train.csv")
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year

plot_sales_trends(df)
plt.show()

print(df["store"].unique())
print(df["country"].unique())
print(df["product"].unique())

dfs_dict = {
    (country, store, product): group
    for (country, store, product), group in df.groupby(["country", "store", "product"])
}

print(dfs_dict.keys())

# %%

for key in dfs_dict.keys():

    try:
        df_analyze = dfs_dict[key]
        df_analyze = df_analyze[["date", "num_sold"]]
        df_analyze = df_analyze.rename(columns={"date": "ds", "num_sold": "y"})

        model = Prophet()
        model.fit(df_analyze)

        # Create a DataFrame for future dates
        future = model.make_future_dataframe(periods=365 * 3, freq="D")

        # Make predictions
        forecast = model.predict(future)

        # Plot the forecast
        plot_plotly(model, forecast).show()

        # Plot components (trend, seasonality, etc.)
        plot_components_plotly(model, forecast).show()
    except ValueError as ve:
        print(f"Error: {ve}")



# %%
