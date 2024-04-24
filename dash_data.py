import pandas as pd
import dash
import dash_core_components as dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/Devarsh01/Data_Viz_Project/main/exported_data.csv")  

# Drop rows with missing values
df_cleaned = df.dropna()

# Parse the 'Genres' column to separate individual categories
df_cleaned['Genres'] = df_cleaned['Genres'].str.split(',')
unique_genres = set(genre for sublist in df_cleaned['Genres'] for genre in sublist)

# Parse the 'Categories' column to separate individual categories
df_cleaned['Categories'] = df_cleaned['Categories'].str.split(',')
unique_categories = set(category for sublist in df_cleaned['Categories'] for category in sublist)

df_cleaned['Date'] = pd.to_datetime(df_cleaned[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1))

df_cleaned['Developers'] = df_cleaned['Developers'].astype(str)

title_style = {'fontname': 'serif', 'color': 'blue', 'size': 'large'}
label_style = {'fontname': 'serif', 'color': 'darkred'}

# Initialize the Dash app
app = dash.Dash(__name__)

main_header = html.Header([
    html.Div([
        html.Img(src="assets/logo.png", style={'height': '50px', 'width': 'auto', 'marginRight': '10px'}),
        html.H1("Video Game Analysis", style={'margin': '0', 'fontSize': '2.5em', 'color': 'navy', 'fontFamily': 'Serif'})
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),  # This div wraps the image and the H1
    html.P("Exploring video game trends and analytics.", style={'textAlign': 'center', 'fontFamily': 'Serif', 'marginTop': '10px'}),
], style={'padding': '1rem', 'backgroundColor': '#f8f8f8'})

# Define the layout of Tab 1
tab1_layout = html.Div([
    html.Div([
        html.H2("Top 5 Games by Revenue", style={**title_style, 'textAlign': 'center'}),
        # Download button for CSV file
         html.Button("Download CSV", id="btn-download-csv", title="Click to download data as CSV"),
    
    # The dcc.Download component
    dcc.Download(id="download-csv"),
    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between', 'margin-bottom': '20px'}),
    dcc.Loading(
        id="loading-graph",
        type="circle",
        children=[
            html.Div([
                html.H4("Select Genres"),
                dcc.Dropdown(
                    id='genre-dropdown',
                    options=[{'label': genre, 'value': genre} for genre in unique_genres],
                    value=['Action'],
                    multi=True,
                    placeholder="Please select a genre"
                )
            ]),
            html.Div([
                html.H4("Price Range"),
                dcc.RangeSlider(
                    id='price-range-slider',
                    min=df_cleaned['Price'].min(),
                    max=df_cleaned['Price'].max(),
                    step=5,
                    marks={i: str(i) for i in range(int(df_cleaned['Price'].min()), int(df_cleaned['Price'].max())+1, 25)},
                    value=[df_cleaned['Price'].min(), df_cleaned['Price'].max()]
                )
            ]),
            dcc.Graph(id='genre-revenue-bar'),
            html.Div(id='table-output')
        ]
    )
])

# Define the layout of Tab 2
tab2_layout = html.Div([
    html.H2("User Base w.r.t Categories", style={**title_style, 'textAlign': 'center'}),
    html.Div([
        dcc.Loading(
            id="loading-graph-categories",
            type="circle",
            children=[
                dcc.Graph(id='category-user-bar', style={'width': '100%', 'height': '50%'}),
            ]
        ),
        dcc.Loading(
            id="loading-pie-chart",
            type="circle",
            children=[
                dcc.Graph(id='pie-chart', style={'width': '100%', 'height': '50%'})
            ]
        )
    ], style={'width': '50%', 'float': 'left', 'margin': '10px 0'}),  # Left half of the page
    html.Div([
        html.H4("Select Categories", style=label_style),
        dcc.Checklist(
            id='category-checklist',
            options=[{'label': category, 'value': category} for category in unique_categories],
            value=['Multi-player'],
            inline=True,
            labelStyle={'display': 'block'},
            style={'textAlign': 'left', 'marginBottom': '10px'}
        ),
        html.H4("Select Operating System", style=label_style),
        dcc.RadioItems(
            id='os-radio',
            options=[
                {'label': 'Windows', 'value': 'Windows'},
                {'label': 'Mac', 'value': 'Mac'},
                {'label': 'Linux', 'value': 'Linux'}
            ],
            value='Windows',  # Default selection
            labelStyle={'display': 'block'}  # Display radio items vertically
        )
    ], style={'width': '50%', 'float': 'left', 'margin': '10px 0'})   # Right half of the page
])

tab3_layout = html.Div([
    html.H2("Games Published Over Time", style={**title_style, 'textAlign': 'center'}),
    html.Div([
        html.H4("Select Date Range", style=label_style),
        dcc.DatePickerRange(
            id='date-range-picker',
            start_date=df_cleaned['Date'].min(),
            end_date=df_cleaned['Date'].max(),
            min_date_allowed=df_cleaned['Date'].min(),
            max_date_allowed=df_cleaned['Date'].max(),
            display_format='YYYY-MM-DD'
        ),
        html.H4("Select Publisher", style=label_style),
        dcc.Dropdown(
            id='developer-publisher-dropdown',
            options=[{'label': name, 'value': name} for name in df_cleaned['Publishers'].unique()],
            value=df_cleaned['Publishers'].unique()[50],
            multi=False
        ),
        html.Div([
            html.H4("Notes", style=label_style),
            dcc.Textarea(
                id='notes-textarea',
                placeholder='Write your notes here...',
                style={'width': '100%', 'height': '25px'}
            )
        ])
    ], style={'width': '30%', 'float': 'left', 'margin': '10px 0'}),
    html.Div([
        dcc.Loading(
            id="loading-line-chart",
            type="circle",
            children=[
                dcc.Graph(id='line-chart', style={'width': '100%', 'height': '50vh'})
            ]
        ),
        dcc.Loading(
            id="loading-review-pie",
            type="circle",
            children=[
                dcc.Graph(id='review-proportion-pie', style={'width': '100%', 'height': '40%'})
            ]
        )
    ], style={'width': '100%', 'float': 'left', 'margin': '10px'}),
])

# Assign the layout to the Dash app
tab_style = {
    'fontWeight': 'bold',
}

app.layout = html.Div(style={'background-image': 'url("/assets/background_image.png")', 'background-size': 'cover'}, children=[
    main_header,  # Include the main header here
    html.Div([
        dcc.Tabs([
            dcc.Tab(label='Genre Analysis', children=tab1_layout, style=tab_style),
            dcc.Tab(label='Category Analysis', children=tab2_layout, style=tab_style),
            dcc.Tab(label='Publisher/Developer Analysis', children=tab3_layout, style=tab_style),
        ])
    ])
])

# Define callback to update the bar plot based on dropdown selection
@app.callback(
    [Output('genre-revenue-bar', 'figure'),
     Output('table-output', 'children')],
    [Input('genre-dropdown', 'value'),
     Input('price-range-slider', 'value')]
)
def update_genre_revenue_bar(selected_genres, price_range):
    if len(selected_genres) < 1:
        return {}, html.Table()
    else:
        # Filter the DataFrame to include only games that belong to all selected genres and fall within the price range
        filtered_df = df_cleaned[
            (df_cleaned['Genres'].apply(lambda x: all(genre in x for genre in selected_genres))) &
            (df_cleaned['Price'] >= price_range[0]) &
            (df_cleaned['Price'] <= price_range[1])
        ]
        
        # Get the top 5 game names based on Revenue for the intersection of selected genres and price range
        top_5_games = filtered_df.nlargest(5, 'Revenue')[['Name', 'Revenue', 'Developers']]
        
        # Plot the top 5 games for the selected genres and price range
        fig = px.bar(top_5_games, x='Name', y='Revenue', title='Top 5 Games by Revenue', color='Revenue', color_continuous_scale='Viridis')
        
        # Generate a table for the top 5 games
        table = html.Table(
        # Header
        [html.Tr([html.Th(col, style={'border': '1px solid #ddd', 'padding': '8px'}) for col in top_5_games.columns])] +
        # Body
        [html.Tr([html.Td(top_5_games.iloc[i][col], style={'border': '1px solid #ddd', 'padding': '8px', 'background-color': 'lightblue'}) for col in top_5_games.columns]) for i in range(len(top_5_games))],
        style={'border-collapse': 'collapse', 'width': '100%'}
    )
        
        return fig, table

@app.callback(
    Output("download-csv", "data"),
    [Input("btn-download-csv", "n_clicks")],
    prevent_initial_call=True,
)
def download_csv(n_clicks):
    # This function converts the DataFrame to a CSV string and sends it as a download
    return dcc.send_data_frame(df_cleaned.to_csv, "my_data.csv", index=False)
    
# Define callback to update the bar plot based on category selection
@app.callback(
    [Output('category-user-bar', 'figure'),
     Output('pie-chart', 'figure')],
    [Input('category-checklist', 'value'),
     Input('os-radio', 'value')]
)
def update_category_user_bar(selected_categories, selected_os):
    if not selected_categories:
        return {}, {}
    else:
        # Convert selected categories to set for easier comparison
        selected_categories_set = set(selected_categories)
        
        # Filter the DataFrame to include only games that have all selected categories and match the selected OS
        filtered_df = df_cleaned[df_cleaned['Categories'].apply(lambda x: selected_categories_set.issubset(set(x))) & (df_cleaned[selected_os] == True)]
        
        # If no games match the selected categories and OS, return empty figures
        if filtered_df.empty:
            return {}, {}
        
        # Plot the top 5 games based on the number of users (Peak CCU)
        top_5_games = filtered_df.nlargest(8, 'Peak CCU')
        bar_fig = px.bar(top_5_games, x='Name', y='Peak CCU', title='Top 5 Games by Selected Categories and OS', color='Peak CCU', color_continuous_scale='Plasma')
        
        # Get the names of games in the bar chart
        bar_chart_game_names = top_5_games['Name'].tolist()
        
        # Filter DataFrame to include only games in the bar chart
        filtered_df_for_pie = filtered_df[filtered_df['Name'].isin(bar_chart_game_names)]
        
        # Calculate the total Peak CCU of the selected games for the pie chart
        total_peak_ccu = filtered_df['Peak CCU'].sum()
        filtered_df_for_pie['Peak CCU Proportion'] = filtered_df_for_pie['Peak CCU'] / total_peak_ccu * 100
        
        # Create the pie chart
        pie_fig = px.pie(filtered_df_for_pie, values='Peak CCU Proportion', names='Name', title='Peak CCU Proportion by Game')
        
        return bar_fig, pie_fig

@app.callback(
    [Output('line-chart', 'figure'),
     Output('review-proportion-pie', 'figure')],
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('developer-publisher-dropdown', 'value')]
)
def update_game_publish_count_and_review_proportion(start_date, end_date, selected_developer):
    # Filter the DataFrame based on the selected date range and developer/publisher
    filtered_df = df_cleaned[(df_cleaned['Date'] >= start_date) & 
                             (df_cleaned['Date'] <= end_date) & 
                             (df_cleaned['Publishers'] == selected_developer)]
    
    # Group the filtered DataFrame by date and count the number of games published each day
    game_publish_count = filtered_df.groupby('Date').size().reset_index(name='Count')
    
    # Create the line chart
    line_chart_fig = px.line(game_publish_count, x='Date', y='Count', title='Games Published Over Time')
    
    # Calculate the total positive and negative reviews for the selected developer
    total_positive_reviews = filtered_df['Positive'].sum()
    total_negative_reviews = filtered_df['Negative'].sum()
    
    # Create the pie chart for review proportion
    review_proportion_fig = px.pie(names=['Positive Reviews', 'Negative Reviews'], values=[total_positive_reviews, total_negative_reviews], title='Proportion of Positive and Negative Reviews')
    
    return line_chart_fig, review_proportion_fig
    
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
