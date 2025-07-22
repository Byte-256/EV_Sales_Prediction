import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class EVMarketAnalyzer:
    def __init__(self, data_path):
        self.raw_data = pd.read_csv(data_path)
        self.sales_data = None
        self.yearly_sales = None
        self.regional_data = None
        self.prepare_data()

    def prepare_data(self):
        """Prepare and clean the data for analysis"""
        # Filter for EV sales data
        self.sales_data = self.raw_data[
            (self.raw_data['parameter'] == "EV sales") &
            (self.raw_data['category'] == "Historical")
        ].copy()

        # Create yearly aggregated data
        self.yearly_sales = self.sales_data.groupby('year')['value'].sum().reset_index()
        self.yearly_sales = self.yearly_sales.sort_values('year').reset_index(drop=True)

        # Create regional data
        self.regional_data = self.sales_data.groupby(['region', 'year'])['value'].sum().reset_index()

        print("✅ Data preparation completed")
        print(f"📊 Years covered: {self.yearly_sales['year'].min()} - {self.yearly_sales['year'].max()}")
        print(f"🌍 Regions covered: {self.sales_data['region'].nunique()}")
        print(f"📈 Total historical sales: {self.sales_data['value'].sum():,.0f}")

    def market_segmentation_analysis(self):
        """Analyze market segments and identify growth patterns"""
        print("\n" + "="*60)
        print("🎯 MARKET SEGMENTATION ANALYSIS")
        print("="*60)

        # Regional market share analysis
        regional_totals = self.sales_data.groupby('region')['value'].sum().reset_index()
        regional_totals = regional_totals.sort_values('value', ascending=False)
        regional_totals['market_share'] = (regional_totals['value'] / regional_totals['value'].sum()) * 100
        regional_totals['cumulative_share'] = regional_totals['market_share'].cumsum()

        # Identify market tiers
        tier1_markets = regional_totals[regional_totals['cumulative_share'] <= 50]['region'].tolist()
        tier2_markets = regional_totals[
            (regional_totals['cumulative_share'] > 50) &
            (regional_totals['cumulative_share'] <= 80)
        ]['region'].tolist()
        tier3_markets = regional_totals[regional_totals['cumulative_share'] > 80]['region'].tolist()

        print(f"🥇 TIER 1 MARKETS (Top 50% market share): {', '.join(tier1_markets)}")
        print(f"🥈 TIER 2 MARKETS (50-80% cumulative): {', '.join(tier2_markets)}")
        print(f"🥉 TIER 3 MARKETS (Emerging): {', '.join(tier3_markets[:10])}")

        # Growth rate analysis by region
        regional_growth = {}
        for region in regional_totals['region'].head(10):  # Top 10 regions
            region_data = self.regional_data[self.regional_data['region'] == region]
            if len(region_data) > 1:
                region_data = region_data.sort_values('year')
                growth_rates = []
                for i in range(1, len(region_data)):
                    if region_data.iloc[i-1]['value'] > 0:
                        growth = ((region_data.iloc[i]['value'] - region_data.iloc[i-1]['value']) /
                                region_data.iloc[i-1]['value']) * 100
                        growth_rates.append(growth)
                if growth_rates:
                    regional_growth[region] = np.mean(growth_rates)

        # Sort regions by growth rate
        growth_sorted = sorted(regional_growth.items(), key=lambda x: x[1], reverse=True)

        print(f"\n📈 TOP GROWTH MARKETS:")
        for region, growth in growth_sorted[:5]:
            print(f"  • {region}: {growth:.1f}% average annual growth")

        return regional_totals, regional_growth

    def competitive_landscape_analysis(self):
        """Analyze competitive landscape and market dynamics"""
        print("\n" + "="*60)
        print("🏆 COMPETITIVE LANDSCAPE ANALYSIS")
        print("="*60)

        # Market concentration analysis (Herfindahl-Hirschman Index)
        regional_totals = self.sales_data.groupby('region')['value'].sum().reset_index()
        regional_totals['market_share'] = (regional_totals['value'] / regional_totals['value'].sum()) * 100
        hhi = (regional_totals['market_share'] ** 2).sum()

        print(f"📊 Market Concentration (HHI): {hhi:.1f}")
        if hhi > 2500:
            concentration_level = "Highly Concentrated"
        elif hhi > 1500:
            concentration_level = "Moderately Concentrated"
        else:
            concentration_level = "Competitive"

        print(f"🎯 Market Structure: {concentration_level}")

        # Market maturity analysis
        recent_years = self.yearly_sales.tail(3)
        recent_growth = []
        for i in range(1, len(recent_years)):
            growth = ((recent_years.iloc[i]['value'] - recent_years.iloc[i-1]['value']) /
                     recent_years.iloc[i-1]['value']) * 100
            recent_growth.append(growth)

        avg_recent_growth = np.mean(recent_growth)

        if avg_recent_growth > 50:
            maturity_stage = "Emerging/High Growth"
        elif avg_recent_growth > 20:
            maturity_stage = "Growth"
        elif avg_recent_growth > 10:
            maturity_stage = "Mature"
        else:
            maturity_stage = "Declining/Saturated"

        print(f"📈 Market Maturity: {maturity_stage} ({avg_recent_growth:.1f}% recent growth)")

        return hhi, concentration_level, maturity_stage

    def seasonal_trend_analysis(self):
        """Analyze seasonal patterns and trends"""
        print("\n" + "="*60)
        print("📅 SEASONAL & TREND ANALYSIS")
        print("="*60)

        # Create time series
        ts_data = self.yearly_sales.set_index('year')['value']

        # Trend analysis
        years = np.array(self.yearly_sales['year'])
        sales = np.array(self.yearly_sales['value'])

        # Fit polynomial trends
        trend_coeffs = np.polyfit(years, sales, 2)  # Quadratic trend
        trend_poly = np.poly1d(trend_coeffs)

        # Calculate trend strength
        trend_values = trend_poly(years)
        trend_strength = np.corrcoef(sales, trend_values)[0, 1]

        print(f"📈 Trend Strength: {trend_strength:.3f}")
        if trend_strength > 0.8:
            trend_description = "Strong upward trend"
        elif trend_strength > 0.5:
            trend_description = "Moderate upward trend"
        else:
            trend_description = "Weak or volatile trend"

        print(f"🎯 Trend Assessment: {trend_description}")

        # Acceleration analysis
        if len(sales) >= 3:
            acceleration = np.diff(np.diff(sales))
            avg_acceleration = np.mean(acceleration)

            if avg_acceleration > 0:
                acceleration_desc = "Accelerating growth"
            elif avg_acceleration < 0:
                acceleration_desc = "Decelerating growth"
            else:
                acceleration_desc = "Steady growth"

            print(f"🚀 Growth Pattern: {acceleration_desc}")

        return trend_strength, trend_description

    def risk_assessment(self):
        """Assess market risks and volatility"""
        print("\n" + "="*60)
        print("⚠️  RISK ASSESSMENT")
        print("="*60)

        # Volatility analysis
        sales_values = self.yearly_sales['value'].values
        returns = np.diff(sales_values) / sales_values[:-1]
        volatility = np.std(returns) * 100

        print(f"📊 Market Volatility: {volatility:.1f}%")

        if volatility > 30:
            risk_level = "High Risk"
        elif volatility > 15:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        print(f"🎯 Risk Level: {risk_level}")

        # Concentration risk (geographic)
        regional_totals = self.sales_data.groupby('region')['value'].sum().reset_index()
        regional_totals['market_share'] = (regional_totals['value'] / regional_totals['value'].sum()) * 100

        top_3_share = regional_totals.nlargest(3, 'market_share')['market_share'].sum()

        if top_3_share > 75:
            concentration_risk = "High concentration risk"
        elif top_3_share > 50:
            concentration_risk = "Medium concentration risk"
        else:
            concentration_risk = "Low concentration risk"

        print(f"🌍 Geographic Concentration: {concentration_risk} ({top_3_share:.1f}% in top 3 markets)")

        return volatility, risk_level, concentration_risk

    def opportunity_identification(self):
        """Identify market opportunities and recommendations"""
        print("\n" + "="*60)
        print("💡 OPPORTUNITY IDENTIFICATION")
        print("="*60)

        # Regional opportunities
        regional_totals = self.sales_data.groupby('region')['value'].sum().reset_index()
        regional_totals = regional_totals.sort_values('value', ascending=False)

        # Calculate market penetration potential
        total_market = regional_totals['value'].sum()

        # Identify underperforming regions with potential
        regional_growth = {}
        for region in regional_totals['region']:
            region_data = self.regional_data[self.regional_data['region'] == region]
            if len(region_data) > 1:
                region_data = region_data.sort_values('year')
                recent_growth = []
                for i in range(max(1, len(region_data)-3), len(region_data)):
                    if i > 0 and region_data.iloc[i-1]['value'] > 0:
                        growth = ((region_data.iloc[i]['value'] - region_data.iloc[i-1]['value']) /
                                region_data.iloc[i-1]['value']) * 100
                        recent_growth.append(growth)
                if recent_growth:
                    regional_growth[region] = np.mean(recent_growth)

        # Score regions (growth potential vs current market size)
        opportunities = []
        for region in regional_totals['region'].head(20):  # Top 20 regions
            current_share = regional_totals[regional_totals['region'] == region]['value'].iloc[0] / total_market
            growth_rate = regional_growth.get(region, 0)

            # Opportunity score: high growth, moderate current share
            opportunity_score = growth_rate * (1 - current_share) * 100
            opportunities.append((region, opportunity_score, growth_rate, current_share))

        # Sort by opportunity score
        opportunities.sort(key=lambda x: x[1], reverse=True)

        print("🎯 TOP MARKET OPPORTUNITIES:")
        for region, score, growth, share in opportunities[:5]:
            print(f"  • {region}: Score {score:.1f} (Growth: {growth:.1f}%, Share: {share:.1%})")

        return opportunities

    def strategic_recommendations(self):
        """Generate strategic recommendations based on analysis"""
        print("\n" + "="*60)
        print("🎯 STRATEGIC RECOMMENDATIONS")
        print("="*60)

        # Market entry strategy
        regional_totals = self.sales_data.groupby('region')['value'].sum().reset_index()
        regional_totals = regional_totals.sort_values('value', ascending=False)

        print("🚀 MARKET ENTRY STRATEGY:")
        print("  1. IMMEDIATE FOCUS:")
        for region in regional_totals['region'].head(3):
            market_size = regional_totals[regional_totals['region'] == region]['value'].iloc[0]
            print(f"     • {region}: {market_size:,.0f} units - Establish/strengthen presence")

        print("\n  2. EXPANSION TARGETS:")
        for region in regional_totals['region'].iloc[3:8]:
            market_size = regional_totals[regional_totals['region'] == region]['value'].iloc[0]
            print(f"     • {region}: {market_size:,.0f} units - Strategic expansion")

        print("\n  3. EMERGING OPPORTUNITIES:")
        for region in regional_totals['region'].iloc[8:13]:
            market_size = regional_totals[regional_totals['region'] == region]['value'].iloc[0]
            print(f"     • {region}: {market_size:,.0f} units - Long-term potential")

        # Investment recommendations
        total_market = self.yearly_sales['value'].sum()
        recent_growth = ((self.yearly_sales['value'].iloc[-1] - self.yearly_sales['value'].iloc[-2]) /
                        self.yearly_sales['value'].iloc[-2]) * 100

        print(f"\n💰 INVESTMENT RECOMMENDATIONS:")
        if recent_growth > 30:
            print("  • AGGRESSIVE EXPANSION: High growth market - increase investment")
            print("  • CAPACITY BUILDING: Scale production and supply chain")
            print("  • TALENT ACQUISITION: Hire for rapid growth")
        elif recent_growth > 15:
            print("  • STRATEGIC GROWTH: Steady expansion with calculated risks")
            print("  • MARKET PENETRATION: Focus on capturing market share")
            print("  • OPERATIONAL EFFICIENCY: Optimize existing operations")
        else:
            print("  • CONSOLIDATION: Focus on profitability and efficiency")
            print("  • MARKET DEFENSE: Protect existing market position")
            print("  • COST OPTIMIZATION: Reduce operational costs")

        print(f"\n🎯 KEY SUCCESS FACTORS:")
        print("  • Product Innovation: Develop next-generation EV technologies")
        print("  • Supply Chain: Secure battery and component supply chains")
        print("  • Infrastructure: Partner with charging infrastructure providers")
        print("  • Regulatory: Monitor and adapt to policy changes")
        print("  • Sustainability: Focus on lifecycle environmental impact")

    def create_comprehensive_dashboard(self):
        """Create comprehensive visualization dashboard"""
        # Create subplot dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Global EV Sales Trend', 'Regional Market Share',
                'Top 10 Markets by Sales', 'Growth Rate Analysis',
                'Market Concentration', 'Forecast vs Reality'
            ],
            specs=[[{"secondary_y": True}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )

        # 1. Global trend
        fig.add_trace(
            go.Scatter(x=self.yearly_sales['year'], y=self.yearly_sales['value'],
                      mode='lines+markers', name='EV Sales'),
            row=1, col=1
        )

        # 2. Regional pie chart
        regional_totals = self.sales_data.groupby('region')['value'].sum().reset_index()
        regional_totals = regional_totals.sort_values('value', ascending=False)

        fig.add_trace(
            go.Pie(labels=regional_totals['region'].head(10),
                   values=regional_totals['value'].head(10),
                   name="Market Share"),
            row=1, col=2
        )

        # 3. Top markets bar chart
        fig.add_trace(
            go.Bar(x=regional_totals['region'].head(10),
                   y=regional_totals['value'].head(10),
                   name="Sales by Region"),
            row=2, col=1
        )

        # 4. Growth rate scatter
        regional_growth = {}
        for region in regional_totals['region'].head(15):
            region_data = self.regional_data[self.regional_data['region'] == region]
            if len(region_data) > 1:
                region_data = region_data.sort_values('year')
                growth_rates = []
                for i in range(1, len(region_data)):
                    if region_data.iloc[i-1]['value'] > 0:
                        growth = ((region_data.iloc[i]['value'] - region_data.iloc[i-1]['value']) /
                                region_data.iloc[i-1]['value']) * 100
                        growth_rates.append(growth)
                if growth_rates:
                    regional_growth[region] = np.mean(growth_rates)

        regions = list(regional_growth.keys())
        growth_rates = list(regional_growth.values())
        market_sizes = [regional_totals[regional_totals['region'] == r]['value'].iloc[0] for r in regions]

        fig.add_trace(
            go.Scatter(x=market_sizes, y=growth_rates,
                      mode='markers', text=regions,
                      name="Growth vs Size"),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(height=1200, showlegend=True)
        fig.update_layout(title_text="EV Market Comprehensive Analysis Dashboard")

        # Show dashboard
        fig.show()

        # Additional static plots
        plt.figure(figsize=(20, 12))

        # Market evolution heatmap
        plt.subplot(2, 3, 1)
        pivot_data = self.regional_data.pivot(index='region', columns='year', values='value')
        pivot_data = pivot_data.fillna(0)
        top_regions = pivot_data.sum(axis=1).nlargest(10).index
        sns.heatmap(pivot_data.loc[top_regions], annot=True, fmt='.0f', cmap='YlOrRd')
        plt.title('Market Evolution by Region')

        # Growth trajectory
        plt.subplot(2, 3, 2)
        for region in top_regions[:5]:
            region_data = self.regional_data[self.regional_data['region'] == region]
            region_data = region_data.sort_values('year')
            plt.plot(region_data['year'], region_data['value'], marker='o', label=region)
        plt.title('Growth Trajectories - Top 5 Markets')
        plt.xlabel('Year')
        plt.ylabel('EV Sales')
        plt.legend()
        plt.grid(True)

        # Market share evolution
        plt.subplot(2, 3, 3)
        yearly_regional = self.regional_data.groupby('year').apply(
            lambda x: x.nlargest(5, 'value')[['region', 'value']]
        ).reset_index(drop=True)

        for region in top_regions[:5]:
            region_share = []
            for year in self.yearly_sales['year']:
                year_data = self.regional_data[self.regional_data['year'] == year]
                total_year = year_data['value'].sum()
                region_year = year_data[year_data['region'] == region]['value'].sum()
                share = (region_year / total_year * 100) if total_year > 0 else 0
                region_share.append(share)
            plt.plot(self.yearly_sales['year'], region_share, marker='o', label=region)

        plt.title('Market Share Evolution (%)')
        plt.xlabel('Year')
        plt.ylabel('Market Share (%)')
        plt.legend()
        plt.grid(True)

        # Volatility analysis
        plt.subplot(2, 3, 4)
        volatility_data = []
        for region in top_regions[:10]:
            region_data = self.regional_data[self.regional_data['region'] == region]
            if len(region_data) > 1:
                region_data = region_data.sort_values('year')
                returns = np.diff(region_data['value']) / region_data['value'].iloc[:-1]
                volatility = np.std(returns) * 100
                volatility_data.append((region, volatility))

        if volatility_data:
            regions, volatilities = zip(*volatility_data)
            plt.barh(regions, volatilities)
            plt.title('Market Volatility by Region')
            plt.xlabel('Volatility (%)')

        # Correlation matrix
        plt.subplot(2, 3, 5)
        correlation_data = pivot_data.loc[top_regions[:8]].T.corr()
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0)
        plt.title('Regional Market Correlations')

        # Future projections
        plt.subplot(2, 3, 6)
        from statsmodels.tsa.arima.model import ARIMA

        # Simple ARIMA forecast for visualization
        ts_data = self.yearly_sales.set_index('year')['value']
        try:
            model = ARIMA(ts_data, order=(1, 1, 1))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=5)

            future_years = range(ts_data.index[-1] + 1, ts_data.index[-1] + 6)

            plt.plot(ts_data.index, ts_data.values, 'o-', label='Historical', color='blue')
            plt.plot(future_years, forecast, 'o-', label='Forecast', color='red')
            plt.title('EV Sales Forecast')
            plt.xlabel('Year')
            plt.ylabel('Sales')
            plt.legend()
            plt.grid(True)
        except:
            plt.text(0.5, 0.5, 'Forecast not available', ha='center', va='center')
            plt.title('EV Sales Forecast')

        plt.tight_layout()
        plt.show()

    def run_comprehensive_analysis(self):
        """Run all analysis components"""
        print("🚀 STARTING COMPREHENSIVE EV MARKET ANALYSIS")
        print("=" * 80)

        # Run all analysis components
        self.market_segmentation_analysis()
        self.competitive_landscape_analysis()
        self.seasonal_trend_analysis()
        self.risk_assessment()
        self.opportunity_identification()
        self.strategic_recommendations()

        # Create visualizations
        print("\n📊 GENERATING COMPREHENSIVE DASHBOARD...")
        self.create_comprehensive_dashboard()

        print("\n✅ ANALYSIS COMPLETE!")
        print("=" * 80)

# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = EVMarketAnalyzer('./data/IEA_Global_EV_Data_2024.csv')

    # Run comprehensive analysis
    analyzer.run_comprehensive_analysis()
