import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import os
import pandas as pd
import io
import base64
from zipfile import ZipFile
from PIL import Image
from model import Detection, crop_image, preprocess_image
from dash import dash_table
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt



# Main app
app = dash.Dash(external_stylesheets=[dbc.themes.LUMEN])
server = app.server
app.config.suppress_callback_exceptions = True


UPLOAD = html.Div([
                    html.Br(),
                    html.H5("Upload an image:", style={'paddingLeft': '1rem', 'display': 'inline-block',"font-weight": "bold"}),
                    html.Br(),
                    dcc.Upload(
                        id='upload-image',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select File')
                        ]),
                        style={
                            #'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        accept=".png, .jpg, .jpeg",
                        # Allow multiple files to be uploaded
                        multiple=False
                    ),
                    html.P("Accepted formats: .png, .jpg, .jpeg", style={'paddingLeft': '1rem', 'display': 'inline-block',"font-weight": "bold", "color": "red"}),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Hr(style={'borderWidth': "0.3vh"}),
                    html.H6("Uploaded Image:", style={'paddingLeft': '1rem', 'display': 'inline-block',"font-weight": "bold"}),
                    html.Br(),
                    dbc.Col(
                                html.Div(id='load-image')
                            )

                ])


QR_IMAGES = html.Div([
                    html.Br(),
                    html.H5("Details of QR codes", style={'paddingLeft': '1rem', 'display': 'inline-block', "font-weight": "bold"}),
                    html.Br(),
                    html.Div([
                        html.Br(),
                        html.Br(),
                        # Create element to hide/show, in this case an 'Input Component'
                        dcc.Input(
                            id = 'element-to-hide',
                            placeholder = 'something',
                            value = 'No QR Code Detected',
                            style={'marginLeft':'30px',
                                   'color': 'red'}
                        )
                    ], style= {'display': 'block','marginLeft':'30px','color': 'red'} # <-- This is the line that will be changed by the dropdown callback
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dash_table.DataTable(id='table_prediction',
                                                    #  columns = [
                                                    #      {'name': 'QR Code', 'id': 'name'},
                                                    #      {'name': 'X', 'id': 'X'},
                                                    #      {'name': 'Y', 'id': 'Y'},
                                                    #      {'name': 'W', 'id': 'W'},
                                                    #      {'name': 'H', 'id': 'H'}
                                                    #  ],
                                                     style_cell={'textAlign': 'center'},
                                                     style_data={
                                                                    'whiteSpace': 'normal',
                                                                    'height': 'auto',
                                                                },
                                                     ),
                                                     style={'margin': '10px'}
                            )
                        ]
                    ),
                    html.Br(),
                    html.H5("Visualization of QR Codes", style={'paddingLeft': '1rem', 'display': 'inline-block', "font-weight": "bold"}),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(id='output-image-crop')
                            )
                        ]
                    )
                    
                    
])




DETECTION = html.Div([
                        html.Br(),
                        dbc.Row([
                            dbc.Col([dbc.Card(UPLOAD, style={"height": "50rem", 'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'background-color':'white'}),
                                html.Br(),
                                ],
                                width={"size":3},
                                style={"margin-left": "20px"}
                                ),
                            dbc.Col([dbc.Card(QR_IMAGES, style={"height": "50rem", 'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'background-color':'white'})],
                                    style={"margin-left": "20px", "margin-right":"20px"})]),
                        html.Br()
                    ])


nav_item = dbc.Nav(
    [
        dbc.NavItem(dbc.NavLink("", href="/analisis"))
    ],
    fill=True
)


nav_menu = dbc.Navbar(
    [
        html.A(
        # Use row and col to control vertical alignment of logo / brand
        dbc.Row(
            [
                dbc.Col(),
                dbc.Col(dbc.NavbarBrand("QR Code Detection", className="m1-2", style={"font-size": "40px" , "fontWeight": "bold"}))
            ],
            align="center"

        ),
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Nav([nav_item], navbar=True, className="'mI -auto" )
    ], 
    color="primary",
    dark=True
)




app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    nav_menu,
    html.Div([
        html.Div([DETECTION], id="detection")
    ],
    style = {'display': 'block'}
    )
])




## CALLBACKS ----------------------------------------------------------------------------------------------------------
@app.callback(Output('table_prediction', 'data'),
              Input('upload-image', 'contents'))
def prediction(content):
    if content is None:
        raise dash.exceptions.PreventUpdate
    content_type, content_string = content.split(',')
    # Decode the base64 string
    content_decoded = base64.b64decode(content_string)
    # Use BytesIO to handle the decoded content
    image = io.BytesIO(content_decoded)
    table_results = Detection(image)

    
    table_results_final = table_results[["QR Code", "X1", "Y1", "X2", "Y2", "QR URL"]]
    table_results_final["X1"] = table_results_final["X1"].astype(int)
    table_results_final["Y1"] = table_results_final["Y1"].astype(int)
    table_results_final["X2"] = table_results_final["X2"].astype(int)
    table_results_final["Y2"] = table_results_final["Y2"].astype(int)
    table_results_final["QR URL"] = np.where(table_results_final["QR URL"]=="", "URL not available", table_results_final["QR URL"])

    return table_results_final.to_dict('records')

@app.callback(Output("load-image", "children"), 
              Input('upload-image', 'contents'))
def qr_image(content):
    if content is None:
        raise dash.exceptions.PreventUpdate
    try:
        content_type, content_string = content.split(',')
        # Decode the base64 string
        content_decoded = base64.b64decode(content_string)
        # Use BytesIO to handle the decoded content
        image = io.BytesIO(content_decoded)
        img = Image.open(image)
        test_image = preprocess_image(img)
        img = np.array(test_image)
    except OSError:
        raise dash.exceptions.PreventUpdate
    
    fig = px.imshow(img, color_continuous_scale="gray")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return dcc.Graph(figure=fig)
    
@app.callback(Output("output-image-crop", "children"), 
              Input('upload-image', 'contents'),
              Input('table_prediction', 'data'))
def qr_image(content, data):
    if content is None:
        raise dash.exceptions.PreventUpdate
    try:
        content_type, content_string = content.split(',')
        # Decode the base64 string
        content_decoded = base64.b64decode(content_string)
        # Use BytesIO to handle the decoded content
        image = io.BytesIO(content_decoded)
        img = Image.open(image)
        test_image = preprocess_image(img)
        img = np.array(test_image)
    except OSError:
        raise dash.exceptions.PreventUpdate
    img_crop = crop_image(img, data)

    if img_crop is None:
        print("No image available")
    else:
        n_images = len(img_crop)
        fig = make_subplots(rows=2, cols=n_images)

        for n, image in enumerate(img_crop):
            fig.add_trace(px.imshow(image).data[0], row=int(n/5)+1, col=n%5+1)
        
        fig.update_layout(coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',width=800,height=800)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        return dcc.Graph(figure=fig)

@app.callback(
   Output(component_id='element-to-hide', component_property='style'),
   [Input('table_prediction', 'data')])

def show_hide_element(table):
   
    if len(table) == 0:
        return {'display': 'block'}
    elif len(table) > 0:
        return {'display': 'none'}

# Main
if __name__ =="__main__":
    app.run_server(debug=True)

