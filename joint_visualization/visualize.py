from bokeh.plotting import figure, ColumnDataSource
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


def embedding(x, method='tsne', perplexity=30):
    if method == 'pca':
        trainer = PCA(n_components=2)
    else:
        trainer = TSNE(n_components=2, perplexity=perplexity)
    return trainer.fit_transform(x)

def initialize_figure(title):
    if title is None:
        title = 'Untitled'

    TOOLTIPS = [
        ("(x,y)", "($x, $y)"),
        ("entity", "@desc"),
    ]

    p = figure(title=title, tooltips=TOOLTIPS)
    p.grid.grid_line_color = None
    p.background_fill_color = "white"
    p.width = 600
    p.height = 600
    return p

def mtext(p, x, y, text, text_color, text_font_size, text_alpha):
    p.text(x, y, text=[text], text_color=text_color, text_align="center",
           text_font_size=text_font_size, text_alpha=text_alpha)

def annotation(p, coordinates, idx_to_text, text_shift=0.05,
    text_font_size='15pt', text_color="firebrick", text_alpha=1.0):

    for idx, text in enumerate(idx_to_text):
        if not text:
            continue
        x_ = coordinates[idx,0] + text_shift
        y_ = coordinates[idx,1] + text_shift
        text = idx_to_text[idx]
        mtext(p, x_, y_, text, text_color, text_font_size, text_alpha)

def draw(coordinates, idx_to_text, p=None, title=None, marker="circle",
         marker_color='orange', marker_size=5, marker_alpha=0.5):

    # prepare figure
    if p is None:
        p = initialize_figure(title)

    # prepare data source
    colors = marker_color if isinstance(marker_color, list) else [marker_color] * coordinates.shape[0]
    source = ColumnDataSource(data=dict(
        x = coordinates[:,0].tolist(),
        y = coordinates[:,1].tolist(),
        desc = idx_to_text,
        fill_color = colors,
    ))

    # scatter plot
    p.scatter('x', 'y', marker=marker, size=marker_size, line_color= 'white',
              fill_color= 'fill_color', alpha=marker_alpha, source=source)
    return p
