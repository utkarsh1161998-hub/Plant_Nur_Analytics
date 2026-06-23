import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducible results
np.random.seed(42)

# ==========================================
# 1. DATA GENERATION (Simulating Nursery Data)
# ==========================================

# Plant Categories & Inventory
plants = ["Monstera", "Snake Plant", "Fiddle Leaf Fig", "Pothos", "Succulent Mix", "Lavender"]
inventory_data = {
    "Plant_ID": [f"P{i:03d}" for i in range(1, 101)],
    "Species": np.random.choice(plants, 100),
    "Stock_Count": np.random.randint(10, 150, 100),
    "Cost_Price": np.random.uniform(5.0, 25.0, 100).round(2),
}
df_inventory = pd.DataFrame(inventory_data)
# Selling price with a standard markup
df_inventory["Retail_Price"] = (df_inventory["Cost_Price"] * 1.8).round(2)

# Simulated IoT Environmental Data (Last 7 days for the greenhouse)
dates = pd.date_range(start="2026-06-16", end="2026-06-22", freq="h")
env_data = {
    "Timestamp": dates,
    "Temperature_C": np.random.normal(loc=24, scale=3, size=len(dates)).round(1),
    "Humidity_Pct": np.random.normal(loc=65, scale=8, size=len(dates)).round(1),
    "Soil_Moisture_Pct": np.random.normal(loc=45, scale=12, size=len(dates)).round(1),
}
df_environmental = pd.DataFrame(env_data)

# Simulated Sales Data (Last 30 days)
sales_dates = pd.date_range(start="2026-05-24", end="2026-06-22", freq="D")
sales_data = {
    "Sale_Date": np.random.choice(sales_dates, 200),
    "Plant_ID": np.random.choice(df_inventory["Plant_ID"], 200),
    "Quantity_Sold": np.random.randint(1, 5, 200),
}
df_sales = pd.DataFrame(sales_data)
# Merge to get pricing information for sales analytics
df_sales = df_sales.merge(df_inventory[["Plant_ID", "Species", "Cost_Price", "Retail_Price"]], on="Plant_ID")
df_sales["Revenue"] = df_sales["Quantity_Sold"] * df_sales["Retail_Price"]
df_sales["Profit"] = df_sales["Revenue"] - (df_sales["Quantity_Sold"] * df_sales["Cost_Price"])


# ==========================================
# 2. ANALYTICS ENGINE
# ==========================================

print("--- NURSERY PERFORMANCE DASHBOARD ---\n")

# A. Inventory Insights
total_plants = df_inventory["Stock_Count"].sum()
total_value = (df_inventory["Stock_Count"] * df_inventory["Retail_Price"]).sum()
print(f"Total Plants in Stock: {total_plants}")
print(f"Total Asset Value (Retail): ${total_value:,.2f}")

# B. Sales & Profitability by Species
species_analytics = df_sales.groupby("Species").agg(
    Units_Sold=("Quantity_Sold", "sum"),
    Total_Revenue=("Revenue", "sum"),
    Total_Profit=("Profit", "sum")
).sort_values(by="Total_Profit", ascending=False)

print("\n[Sales & Profit Breakdown by Species]")
print(species_analytics)

# C. Environmental Health Check (Alerts)
low_moisture_events = df_environmental[df_environmental["Soil_Moisture_Pct"] < 30]
print(f"\n[IoT Climate Alert]")
print(f"Critical Low Soil Moisture Events (<30%) recorded: {len(low_moisture_events)} hours.")
if len(low_moisture_events) > 0:
    print("👉 Action Required: Check automated irrigation systems.")


# ==========================================
# 3. VISUALIZATIONS
# ==========================================

# Plot 1: Revenue vs Profit by Plant Species
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
species_analytics[["Total_Revenue", "Total_Profit"]].plot(kind="bar", ax=plt.gca(), color=["#2e7d32", "#81c784"])
plt.title("Financial Performance by Species")
plt.ylabel("USD ($)")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Plot 2: Environmental Trends (Hourly over last 7 days)
plt.subplot(1, 2, 2)
# Resampling to Daily average for a cleaner line plot
df_daily_env = df_environmental.resample("D", on="Timestamp").mean()
plt.plot(df_daily_env.index, df_daily_env["Temperature_C"], label="Avg Temp (°C)", color="#e65100", marker="o")
plt.plot(df_daily_env.index, df_daily_env["Soil_Moisture_Pct"], label="Avg Soil Moisture (%)", color="#0288d1", marker="s")
plt.title("7-Day Greenhouse Climate Trends")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
