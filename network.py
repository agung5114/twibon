import pandas as pd
import networkx as nx
from networkx.drawing.layout import spring_layout,random_layout
import plotly.graph_objects as go

def draw_network(df,Source,Target,nodecol,edgecol):
    G = nx.from_pandas_edgelist(df,Source,Target,create_using=nx.Graph)
    # P = spring_layout(G)
    # nx.set_node_attributes(G, pd.Series(edge1.Sentiment, index=edge1.Sentiment).to_dict(), 'Sentiment')
    P = random_layout(G)
    nx.set_node_attributes(G, P, 'pos')
    deg = dict(G.degree)
    node_size= [v +10 if v<100 else 100+(v*10/100) for v in deg.values()]
    edge_x = []
    edge_y = []
    for edge in G.edges():
        print(G.nodes[edge[0]]['pos'])
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        hoverinfo='none',
        mode='lines',
        line=dict(width=0.5, color=edgecol))

    #nodes
    node_x = []
    node_y = []
    node_z = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        z = [node,G.degree(node)]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        text=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale=nodecol,
            reversescale=True,
            color= node_size,
            # color= color_p,
            # size = node_size,
            line_width=2,))

    node_trace.marker.size = node_size
    return edge_trace,node_trace

def networkFig(df,Source,Target,nettitle):
    net = draw_network(df, Source, Target, 'Burg', 'teal')
    netfig = go.Figure(data=[net[0], net[1]],
                layout=go.Layout(
                title=nettitle,
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return netfig