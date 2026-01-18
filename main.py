import boto3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

# AWS Configuration
REGION = "us-east-1"
LAMBDA_FUNCTION_NAME = "brohood-chatbot"
API_NAME = "BroHood-Chatbot-API"  # Your API Gateway name
API_ID = "xq0p7cwr3d"  # Will be auto-discovered from API_NAME
API_STAGE = "prod"

# Bedrock Model IDs to track
BEDROCK_MODELS = [
    "meta.llama3-70b-instruct-v1:0",
    "meta.llama3-8b-instruct-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0"
]

# Time range for metrics
HOURS_BACK = 72  # Last 24 hours
PERIOD = 300  # 5-minute intervals

print("="*70)
print("📊 BroHood Chatbot - AWS Metrics Collection")
print("="*70)
print(f"📍 Region: {REGION}")
print(f"🔧 Lambda: {LAMBDA_FUNCTION_NAME}")
print(f"🌐 API Gateway: {API_NAME}")
print(f"📅 Collecting data for last {HOURS_BACK} hours")
print("="*70 + "\n")

# ==================== INITIALIZE AWS CLIENTS ====================

print("🔌 Initializing AWS clients...")
cloudwatch = boto3.client("cloudwatch", region_name=REGION)
cost_explorer = boto3.client("ce", region_name="us-east-1")  # Cost Explorer is always us-east-1
apigateway = boto3.client("apigateway", region_name=REGION)

# Auto-discover API ID from API name if not provided
if API_ID is None and API_NAME:
    print(f"🔍 Looking for API Gateway: {API_NAME}...")
    try:
        apis_response = apigateway.get_rest_apis(limit=500)
        for api in apis_response.get('items', []):
            if api['name'] == API_NAME:
                API_ID = api['id']
                print(f"✅ Found API ID: {API_ID}")
                break
        
        if API_ID is None:
            print(f"⚠️ API '{API_NAME}' not found. Available APIs:")
            for api in apis_response.get('items', []):
                print(f"   - {api['name']} (ID: {api['id']})")
    except Exception as e:
        print(f"⚠️ Error discovering API: {str(e)}")

# Define time range
END_TIME = datetime.utcnow()
START_TIME = END_TIME - timedelta(hours=HOURS_BACK)

print(f"✅ CloudWatch client initialized")
print(f"✅ Cost Explorer client initialized")
print(f"📅 Start: {START_TIME.strftime('%Y-%m-%d %H:%M:%S')} UTC")
print(f"📅 End:   {END_TIME.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")

# ==================== HELPER FUNCTION ====================

def fetch_cloudwatch_metrics(namespace, metric_names, dimensions, stats):
    """
    Fetch metrics from CloudWatch for a given namespace and dimensions.
    
    Args:
        namespace: AWS service namespace (e.g., 'AWS/Lambda')
        metric_names: List of metric names to fetch
        dimensions: List of dimension dicts for filtering
        stats: Statistics to retrieve (e.g., ['Sum', 'Average'])
    
    Returns:
        List of metric records
    """
    records = []
    
    for metric in metric_names:
        try:
            response = cloudwatch.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric,
                Dimensions=dimensions,
                StartTime=START_TIME,
                EndTime=END_TIME,
                Period=PERIOD,
                Statistics=stats
            )
            
            for point in response.get("Datapoints", []):
                record = {
                    "service": namespace,
                    "metric": metric,
                    "timestamp": point["Timestamp"]
                }
                
                for stat in stats:
                    if stat in point:
                        record[stat.lower()] = point[stat]
                
                records.append(record)
        
        except Exception as e:
            print(f"⚠️ Error fetching {metric}: {str(e)}")
    
    return records

# ==================== COLLECT BEDROCK METRICS ====================

print("🤖 Collecting Bedrock metrics...")
bedrock_metrics = [
    "InvocationLatency",
    "InputTokenCount",
    "OutputTokenCount"
]


all_bedrock_data = []

for model_id in BEDROCK_MODELS:
    print(f"  📊 Fetching metrics for {model_id.split('/')[-1]}...")
    
    bedrock_data = fetch_cloudwatch_metrics(
        namespace="AWS/Bedrock",
        metric_names=bedrock_metrics,
        dimensions=[{"Name": "ModelId", "Value": model_id}],
          stats=["Sum", "Average"]
    )
    
    # Add model name to each record
    for record in bedrock_data:
        record['model'] = model_id.split(':')[0]  # Keep full model identifier
    
    all_bedrock_data.extend(bedrock_data)

df_bedrock = pd.DataFrame(all_bedrock_data)

if not df_bedrock.empty:
    df_bedrock['timestamp'] = pd.to_datetime(df_bedrock['timestamp'])
    df_bedrock = df_bedrock.sort_values('timestamp')
    print(f"✅ Collected {len(df_bedrock)} Bedrock metric datapoints")
    print(f"   Metrics: {df_bedrock['metric'].unique().tolist()}\n")
else:
    print("⚠️ No Bedrock metrics found (data may not be available yet)\n")

# ==================== COLLECT LAMBDA METRICS ====================

print("⚡ Collecting Lambda metrics...")
lambda_metrics = ["Invocations", "Duration", "Errors", "Throttles", "ConcurrentExecutions"]

lambda_data = fetch_cloudwatch_metrics(
    namespace="AWS/Lambda",
    metric_names=lambda_metrics,
    dimensions=[{"Name": "FunctionName", "Value": LAMBDA_FUNCTION_NAME}],
    stats=["Sum", "Average", "Maximum"]
)

df_lambda = pd.DataFrame(lambda_data)

if not df_lambda.empty:
    df_lambda['timestamp'] = pd.to_datetime(df_lambda['timestamp'])
    df_lambda = df_lambda.sort_values('timestamp')
    print(f"✅ Collected {len(df_lambda)} Lambda metric datapoints")
    print(f"   Metrics: {df_lambda['metric'].unique().tolist()}\n")
else:
    print("⚠️ No Lambda metrics found\n")

# ==================== COLLECT API GATEWAY METRICS ====================

print("🌐 Collecting API Gateway metrics...")

if API_ID is None:
    print(f"⚠️ API ID not found - skipping API Gateway metrics\n")
    api_data = []
else:
    api_metrics = ["Count", "Latency", "IntegrationLatency", "4XXError", "5XXError"]
    
    # Try with ApiName and Stage first (more common)
    print(f"  📊 Trying ApiName={API_NAME}, Stage={API_STAGE}...")
    api_data = fetch_cloudwatch_metrics(
        namespace="AWS/ApiGateway",
        metric_names=api_metrics,
        dimensions=[
            {"Name": "ApiName", "Value": API_NAME},
            {"Name": "Stage", "Value": API_STAGE}
        ],
        stats=["Sum", "Average", "Maximum"]
    )
    
    # If no data, try with just ApiName
    if not api_data:
        print(f"  📊 Trying ApiName={API_NAME} only...")
        api_data = fetch_cloudwatch_metrics(
            namespace="AWS/ApiGateway",
            metric_names=api_metrics,
            dimensions=[
                {"Name": "ApiName", "Value": API_NAME}
            ],
            stats=["Sum", "Average", "Maximum"]
        )
    
    # If still no data, try with ApiId and Stage
    if not api_data:
        print(f"  📊 Trying ApiId={API_ID}, Stage={API_STAGE}...")
        api_data = fetch_cloudwatch_metrics(
            namespace="AWS/ApiGateway",
            metric_names=api_metrics,
            dimensions=[
                {"Name": "ApiId", "Value": API_ID},
                {"Name": "Stage", "Value": API_STAGE}
            ],
            stats=["Sum", "Average", "Maximum"]
        )
    
    # Last resort: try without any dimensions (account-level)
    if not api_data:
        print(f"  📊 Trying account-level metrics...")
        api_data = fetch_cloudwatch_metrics(
            namespace="AWS/ApiGateway",
            metric_names=["Count"],  # Just try Count metric
            dimensions=[],
            stats=["Sum"]
        )

df_api = pd.DataFrame(api_data)

if not df_api.empty:
    df_api['timestamp'] = pd.to_datetime(df_api['timestamp'])
    df_api = df_api.sort_values('timestamp')
    print(f"✅ Collected {len(df_api)} API Gateway metric datapoints")
    print(f"   Metrics: {df_api['metric'].unique().tolist()}\n")
else:
    print("⚠️ No API Gateway metrics found\n")

# ==================== COLLECT COST METRICS ====================

print("💰 Collecting cost metrics...")
print("   Note: Cost data typically appears 6-24 hours after usage")

try:
    cost_response = cost_explorer.get_cost_and_usage(
        TimePeriod={
            "Start": (END_TIME - timedelta(days=7)).strftime("%Y-%m-%d"),
            "End": END_TIME.strftime("%Y-%m-%d")
        },
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
        Filter={
            "Dimensions": {
                "Key": "SERVICE",
                "Values": ["Amazon Bedrock", "AWS Lambda", "Amazon API Gateway"]
            }
        },
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}]
    )
    
    cost_records = []
    for result in cost_response["ResultsByTime"]:
        date = result["TimePeriod"]["Start"]
        for group in result["Groups"]:
            service = group["Keys"][0]
            cost = float(group["Metrics"]["UnblendedCost"]["Amount"])
            cost_records.append({
                "date": date,
                "service": service,
                "cost_usd": cost
            })
    
    df_cost = pd.DataFrame(cost_records)
    
    if not df_cost.empty:
        df_cost['date'] = pd.to_datetime(df_cost['date'])
        print(f"✅ Collected cost data for {len(df_cost)} days")
        print(f"   Total cost: ${df_cost['cost_usd'].sum():.4f}\n")
    else:
        print("⚠️ Cost data available but empty\n")

except Exception as e:
    df_cost = pd.DataFrame()
    print(f"⚠️ Cost data not available yet: {str(e)}\n")

# ==================== SUMMARY STATISTICS ====================

print("="*70)
print("📊 SUMMARY STATISTICS (Last 24 Hours)")
print("="*70)

# Lambda Summary
if not df_lambda.empty:
    print("\n⚡ LAMBDA FUNCTION:")
    inv = df_lambda[df_lambda['metric'] == 'Invocations']['sum'].sum()
    avg_duration = df_lambda[df_lambda['metric'] == 'Duration']['average'].mean()
    errors = df_lambda[df_lambda['metric'] == 'Errors']['sum'].sum()
    
    print(f"  Total Invocations: {int(inv)}")
    print(f"  Avg Duration: {avg_duration:.2f} ms")
    print(f"  Total Errors: {int(errors)}")
    print(f"  Error Rate: {(errors/inv*100 if inv > 0 else 0):.2f}%")

# API Gateway Summary
if not df_api.empty:
    print("\n🌐 API GATEWAY:")
    requests = df_api[df_api['metric'] == 'Count']['sum'].sum()
    avg_latency = df_api[df_api['metric'] == 'Latency']['average'].mean()
    errors_4xx = df_api[df_api['metric'] == '4XXError']['sum'].sum()
    errors_5xx = df_api[df_api['metric'] == '5XXError']['sum'].sum()
    
    print(f"  Total Requests: {int(requests)}")
    print(f"  Avg Latency: {avg_latency:.2f} ms")
    print(f"  4XX Errors: {int(errors_4xx)}")
    print(f"  5XX Errors: {int(errors_5xx)}")

# Bedrock Summary
if not df_bedrock.empty:
    print("\n🤖 BEDROCK MODELS:")
    # Calculate invocations from token data (each request has tokens)
    input_token_records = df_bedrock[df_bedrock['metric'] == 'InputTokenCount']
    inv = len(input_token_records[input_token_records['sum'] > 0]) if not input_token_records.empty else 0
    
    input_tokens = df_bedrock[df_bedrock['metric'] == 'InputTokenCount']['sum'].sum()
    output_tokens = df_bedrock[df_bedrock['metric'] == 'OutputTokenCount']['sum'].sum()
    
    print(f"  Total Invocations: {int(inv)}")
    print(f"  Total Input Tokens: {int(input_tokens):,}")
    print(f"  Total Output Tokens: {int(output_tokens):,}")
    print(f"  Avg Tokens per Request: {int((input_tokens + output_tokens)/inv if inv > 0 else 0):,}")

# Cost Summary
if not df_cost.empty:
    print("\n💰 COSTS:")
    print(f"  Total (Last 7 days): ${df_cost['cost_usd'].sum():.4f}")
    print(f"  Avg Daily Cost: ${df_cost.groupby('date')['cost_usd'].sum().mean():.4f}")

print("\n" + "="*70)

# ==================== VISUALIZATIONS ====================

print("\n📈 Generating visualizations...\n")

# Bedrock Metrics Visualization
if not df_bedrock.empty:
    print("📊 Creating Bedrock metrics visualization...")
    
    # Create aggregated data by model and metric
    # Calculate invocations from InputTokenCount (each request has input tokens)
    inv_data = df_bedrock[df_bedrock['metric'] == 'InputTokenCount'].copy()
    inv_data['invocations'] = (inv_data['sum'] > 0).astype(int)  # Count requests
    
    lat_data = df_bedrock[df_bedrock['metric'] == 'InvocationLatency']
    in_tokens = df_bedrock[df_bedrock['metric'] == 'InputTokenCount']
    out_tokens = df_bedrock[df_bedrock['metric'] == 'OutputTokenCount']
    
    # Check if we have any data for each metric
    has_invocations = not inv_data.empty
    has_latency = not lat_data.empty
    has_input_tokens = not in_tokens.empty
    has_output_tokens = not out_tokens.empty
    
    if has_invocations or has_latency or has_input_tokens or has_output_tokens:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Model Invocations", "Average Latency (ms)", "Input Tokens", "Output Tokens")
        )
        
        # 1. Invocations over time (from token data)
        if has_invocations:
            for model in inv_data['model'].unique():
                model_data = inv_data[inv_data['model'] == model].copy()
                # Count requests per timestamp
                agg_data = model_data.groupby('timestamp').agg({'sum': 'count'}).reset_index()
                agg_data.rename(columns={'sum': 'invocations'}, inplace=True)
                fig.add_trace(
                    go.Scatter(x=agg_data['timestamp'], y=agg_data['invocations'],
                              name=model, mode='lines+markers'),
                    row=1, col=1
                )
        
        # 2. Latency over time
        if has_latency:
            for model in lat_data['model'].unique():
                model_data = lat_data[lat_data['model'] == model].copy()
                agg_data = model_data.groupby('timestamp').agg({'average': 'mean'}).reset_index()
                fig.add_trace(
                    go.Scatter(x=agg_data['timestamp'], y=agg_data['average'],
                              name=model, mode='lines', showlegend=False),
                    row=1, col=2
                )
        
        # 3. Input Tokens over time
        if has_input_tokens:
            for model in in_tokens['model'].unique():
                model_data = in_tokens[in_tokens['model'] == model].copy()
                agg_data = model_data.groupby('timestamp').agg({'sum': 'sum'}).reset_index()
                fig.add_trace(
                    go.Scatter(x=agg_data['timestamp'], y=agg_data['sum'],
                              name=model, mode='lines', showlegend=False),
                    row=2, col=1
                )
        
        # 4. Output Tokens over time
        if has_output_tokens:
            for model in out_tokens['model'].unique():
                model_data = out_tokens[out_tokens['model'] == model].copy()
                agg_data = model_data.groupby('timestamp').agg({'sum': 'sum'}).reset_index()
                fig.add_trace(
                    go.Scatter(x=agg_data['timestamp'], y=agg_data['sum'],
                              name=model, mode='lines', showlegend=False),
                    row=2, col=2
                )
        
        fig.update_layout(height=700, title_text="🤖 Bedrock Model Metrics", showlegend=True)
        fig.show()
        
        # Create a summary bar chart by model
        print("📊 Creating Bedrock summary by model...")
        summary_data = []
        for model in df_bedrock['model'].unique():
            model_df = df_bedrock[df_bedrock['model'] == model]
            # Calculate invocations from input token records
            inv_count = len(model_df[model_df['metric'] == 'InputTokenCount']) if has_invocations else 0
            avg_lat = model_df[model_df['metric'] == 'InvocationLatency']['average'].mean() if has_latency else 0
            in_tok = model_df[model_df['metric'] == 'InputTokenCount']['sum'].sum() if has_input_tokens else 0
            out_tok = model_df[model_df['metric'] == 'OutputTokenCount']['sum'].sum() if has_output_tokens else 0
            
            summary_data.append({
                'model': model,
                'invocations': inv_count,
                'avg_latency_ms': avg_lat,
                'input_tokens': in_tok,
                'output_tokens': out_tok,
                'total_tokens': in_tok + out_tok
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Bar charts for summary
        fig2 = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Total Invocations", "Avg Latency (ms)", "Total Tokens")
        )
        
        fig2.add_trace(
            go.Bar(x=df_summary['model'], y=df_summary['invocations'], name='Invocations'),
            row=1, col=1
        )
        
        fig2.add_trace(
            go.Bar(x=df_summary['model'], y=df_summary['avg_latency_ms'], name='Latency', marker_color='orange'),
            row=1, col=2
        )
        
        fig2.add_trace(
            go.Bar(x=df_summary['model'], y=df_summary['total_tokens'], name='Tokens', marker_color='green'),
            row=1, col=3
        )
        
        fig2.update_layout(height=400, title_text="📊 Bedrock Summary Statistics", showlegend=False)
        fig2.show()
    else:
        print("⚠️ No Bedrock metrics available for visualization")
else:
    print("⚠️ No Bedrock data to visualize")

# Lambda Function Metrics
if not df_lambda.empty:
    print("📊 Creating Lambda metrics visualization...")
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Invocations", "Duration (ms)", "Errors", "Throttles")
    )
    
    # 1. Invocations
    inv_data = df_lambda[df_lambda['metric'] == 'Invocations']
    if not inv_data.empty:
        fig.add_trace(
            go.Scatter(x=inv_data['timestamp'], y=inv_data['sum'],
                      name='Invocations', mode='lines+markers', fill='tozeroy'),
            row=1, col=1
        )
    
    # 2. Duration
    dur_data = df_lambda[df_lambda['metric'] == 'Duration']
    if not dur_data.empty:
        fig.add_trace(
            go.Scatter(x=dur_data['timestamp'], y=dur_data['average'],
                      name='Avg Duration', mode='lines', line=dict(color='orange')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=dur_data['timestamp'], y=dur_data['maximum'],
                      name='Max Duration', mode='lines', line=dict(dash='dot', color='red')),
            row=1, col=2
        )
    
    # 3. Errors
    err_data = df_lambda[df_lambda['metric'] == 'Errors']
    if not err_data.empty:
        fig.add_trace(
            go.Bar(x=err_data['timestamp'], y=err_data['sum'],
                   name='Errors', marker_color='red'),
            row=2, col=1
        )
    
    # 4. Throttles
    thr_data = df_lambda[df_lambda['metric'] == 'Throttles']
    if not thr_data.empty:
        fig.add_trace(
            go.Bar(x=thr_data['timestamp'], y=thr_data['sum'],
                   name='Throttles', marker_color='purple'),
            row=2, col=2
        )
    
    fig.update_layout(height=700, title_text="⚡ Lambda Function Performance", showlegend=True)
    fig.show()
else:
    print("⚠️ No Lambda data to visualize")

# API Gateway Metrics
if not df_api.empty:
    print("📊 Creating API Gateway metrics visualization...")
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Request Count", "Latency (ms)", "4XX Errors", "5XX Errors")
    )
    
    # 1. Request Count
    count_data = df_api[df_api['metric'] == 'Count']
    if not count_data.empty:
        fig.add_trace(
            go.Scatter(x=count_data['timestamp'], y=count_data['sum'],
                      name='Requests', mode='lines+markers', fill='tozeroy',
                      line=dict(color='green')),
            row=1, col=1
        )
    
    # 2. Latency
    lat_data = df_api[df_api['metric'] == 'Latency']
    int_lat_data = df_api[df_api['metric'] == 'IntegrationLatency']
    
    if not lat_data.empty:
        fig.add_trace(
            go.Scatter(x=lat_data['timestamp'], y=lat_data['average'],
                      name='Total Latency', mode='lines'),
            row=1, col=2
        )
    
    if not int_lat_data.empty:
        fig.add_trace(
            go.Scatter(x=int_lat_data['timestamp'], y=int_lat_data['average'],
                      name='Integration Latency', mode='lines', line=dict(dash='dot')),
            row=1, col=2
        )
    
    # 3. 4XX Errors
    err4_data = df_api[df_api['metric'] == '4XXError']
    if not err4_data.empty:
        fig.add_trace(
            go.Bar(x=err4_data['timestamp'], y=err4_data['sum'],
                   name='4XX Errors', marker_color='orange'),
            row=2, col=1
        )
    
    # 4. 5XX Errors
    err5_data = df_api[df_api['metric'] == '5XXError']
    if not err5_data.empty:
        fig.add_trace(
            go.Bar(x=err5_data['timestamp'], y=err5_data['sum'],
                   name='5XX Errors', marker_color='red'),
            row=2, col=2
        )
    
    fig.update_layout(height=700, title_text="🌐 API Gateway Metrics", showlegend=True)
    fig.show()
else:
    print("⚠️ No API Gateway data to visualize")

# Cost Analysis
if not df_cost.empty:
    print("📊 Creating cost analysis visualization...")
    
    # Daily costs by service
    fig = px.bar(df_cost, x='date', y='cost_usd', color='service',
                 title='💰 Daily AWS Costs by Service',
                 labels={'cost_usd': 'Cost (USD)', 'date': 'Date'},
                 barmode='group')
    
    fig.update_layout(height=400)
    fig.show()
    
    # Total cost summary
    total_by_service = df_cost.groupby('service')['cost_usd'].sum().reset_index()
    
    fig2 = px.pie(total_by_service, values='cost_usd', names='service',
                  title='Total Cost Distribution')
    fig2.update_layout(height=400)
    fig2.show()
else:
    print("⚠️ No cost data available (data typically appears after 6-24 hours)")

# ==================== EXPORT TO CSV ====================

print("\n💾 Exporting data to CSV files...")

if not df_bedrock.empty:
    df_bedrock.to_csv('bedrock_metrics.csv', index=False)
    print("✅ Saved: bedrock_metrics.csv")

if not df_lambda.empty:
    df_lambda.to_csv('lambda_metrics.csv', index=False)
    print("✅ Saved: lambda_metrics.csv")

if not df_api.empty:
    df_api.to_csv('api_gateway_metrics.csv', index=False)
    print("✅ Saved: api_gateway_metrics.csv")

if not df_cost.empty:
    df_cost.to_csv('cost_metrics.csv', index=False)
    print("✅ Saved: cost_metrics.csv")

print("\n🎉 Data collection and visualization complete!")
print("="*70)