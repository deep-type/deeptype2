import numpy as np
from itertools import product
from matplotlib.colors import LinearSegmentedColormap

Blues = LinearSegmentedColormap("Blues",
    {
        'blue': [(0.0, 1.0, 1.0),
                  (0.125, 0.9686274528503418, 0.9686274528503418),
                  (0.25, 0.93725490570068359, 0.93725490570068359),
                  (0.375, 0.88235294818878174, 0.88235294818878174),
                  (0.5, 0.83921569585800171, 0.83921569585800171),
                  (0.625, 0.7764706015586853, 0.7764706015586853),
                  (0.75, 0.70980393886566162, 0.70980393886566162),
                  (0.875, 0.61176472902297974, 0.61176472902297974),
                  (1.0, 0.41960784792900085, 0.41960784792900085)],
        'green': [(0.0, 0.9843137264251709, 0.9843137264251709),
                  (0.125, 0.92156863212585449, 0.92156863212585449),
                  (0.25, 0.85882353782653809, 0.85882353782653809),
                  (0.375, 0.7921568751335144, 0.7921568751335144),
                  (0.5, 0.68235296010971069, 0.68235296010971069),
                  (0.625, 0.57254904508590698, 0.57254904508590698),
                  (0.75, 0.44313725829124451, 0.44313725829124451),
                  (0.875, 0.31764706969261169, 0.31764706969261169),
                  (1.0, 0.18823529779911041, 0.18823529779911041)],
        'red': [(0.0, 0.9686274528503418, 0.9686274528503418),
                (0.125, 0.87058824300765991, 0.87058824300765991),
                (0.25, 0.7764706015586853, 0.7764706015586853),
                (0.375, 0.61960786581039429, 0.61960786581039429),
                (0.5, 0.41960784792900085, 0.41960784792900085),
                (0.625, 0.25882354378700256, 0.25882354378700256),
                (0.75, 0.12941177189350128, 0.12941177189350128),
                (0.875, 0.031372550874948502, 0.031372550874948502),
                (1.0, 0.031372550874948502, 0.031372550874948502)]
    }
)



def th_cell(name, rotated=False, rowspan=None):
    rowspan = "" if rowspan is None else "rowspan='%d'" % (rowspan,)
    return "<th class='{classname}' {rowspan}><div><span>{name}</span></div></th>".format(
        name=name,
        classname="rotate" if rotated else "default",
        rowspan=rowspan
    )

def html_cell(name, color, textcolor):
    return "<td class='cell' style='background:rgb({red},{green},{blue});color:{textcolor}'>{name}</td>".format(
        red=int(255 * (color[0])),
        green=int(255 * (color[1])),
        blue=int(255 * (color[2])),
        name=name,
        textcolor=textcolor
    )


def get_textcolor(alpha, denominator):
    if denominator == 0:
        return "#333"
    if alpha > 0.5:
        return "white"
    if alpha < 1e-6:
        return "rgb(100, 100, 100)"
    return "black"


def html_row(counts, names, cmap):
    denom = counts.sum()
    if denom == 0:
        denom = 1e-6
    return "".join([html_cell("%.2f" % (counts[i] / denom), color=cmap(counts[i] / denom),
                 textcolor=get_textcolor(counts[i] / denom, denom))
                 for i in range(len(names))])

def plot_confusion_matrix(cm, classes, savename, title, figsize, cmap=None,
                          xlabel="Predicted label", ylabel="True label"):
    """
    This function creates an html-based confusion matrix.
    """
    headers, rows = [], []
    headers.append(
        "<tr><th></th><th></th><th colspan='{colspan}'>{xlabel}</th></tr>".format(
            colspan=len(classes),
            xlabel=xlabel
        )
    )
    headers.append(
        "<tr>" + "".join([th_cell("", rotated=False), th_cell("", rotated=False)] + [th_cell(cname, rotated=True) for cname in classes]) + "</tr>"
    )
    if cmap is None:
        cmap = Blues
    label_presence = cm.sum(axis=1)
    num_effective_rows = int((label_presence > 0).sum())
    added_tr = False
    for i, cname in enumerate(classes):
        if label_presence[i] > 0:
          row_tds = th_cell(cname) + html_row(cm[i], classes, cmap=cmap)
          if not added_tr:
              row_tds = th_cell(ylabel, rowspan=num_effective_rows) + row_tds
              added_tr = True
          rows.append("<tr>" + row_tds + "</tr>")
    rows = "\n".join(rows)
    headers = "\n".join(headers)
    html = """
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
        </head>
        <body>
            <style>{style}</style>
            <div>
                <h2>{title}</h2>
            </div>
            <table class='table'>
                <thead>
                    {headers}
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </body>
    </html>
    """.format(
        title=title,
        rows=rows,
        headers=headers,
        style="""
        .table {border-collapse: collapse;}
        .cell {width: 5px;height: 5px;text-align:right; font-size: 8px;}
        th.default {
            text-align:right;
            padding-right: 2px;
        }
        th.rotate {
          /* Something you can count on */
          height: 140px;
          white-space: nowrap;
          text-align:left;
        }
        th.rotate > div {
          -webkit-transform: translate(16px, 51px) rotate(315deg);
          transform: translate(16px, 51px) rotate(315deg);
          width: 30px;
        }
        th.rotate > div > span {
          border-bottom: 1px solid #ccc;
          padding: 5px 10px;
        }
        """
    )
    with open(savename, "wt") as fout:
        fout.write(html)


def matplotlib_plot_confusion_matrix(cm,
                                     classes,
                                     savename,
                                     figsize=(20, 20),
                                     normalize=True,
                                     title='Confusion matrix',
                                     cmap=None,
                                     xlabel="Predicted label",
                                     ylabel="True label"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if cmap is None:
        cmap = plt.cm.Blues
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    not_nan = np.logical_not(np.isnan(cm))
    if not_nan.sum() > 0:
        thresh = cm[not_nan].max() / 2.
    else:
        thresh = 0.5

    res = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    cb = plt.colorbar(res)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.annotate("%.2f" % (cm[i, j],), xy=(j, i),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    import time
    plt.tight_layout()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(savename, bbox_inches="tight")
    plt.close()
