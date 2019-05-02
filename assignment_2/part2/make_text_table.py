import pandas as pd
from argparse import ArgumentParser


def make_table(fpath, steps):
    df = pd.read_csv(fpath, sep=';', encoding='iso-8859-1', names=['step', 'length', 'temp', 'text'])
    lengths = sorted(df.length.unique())
    temps = sorted(df.temp.unique())

    s_multirow = "\\multirow{{{}}}{{*}}{{{}}}"
    s_text = "& {} & \\makecell{{{}}}\\\\"

    output = ""
    for idx, step in enumerate(steps):
        sdf = df[df.step == step]
        # IF FULL BLOCK
        if idx % 3 == 0:
            if idx != 0:
                output += "}\n\\par"
            output += "\n\\hspace*{-0.2\\textwidth}\\resizebox{1.4\\textwidth}{!}{\n"
        # BEGIN TABLE
        output += "\\begin{minipage}{0.7\\textwidth}\n\\begin{tabularx}{\\textwidth}{ccX}\n"
        output += "\\multicolumn{3}{c}{\\Large {" + str(step) + "}}\\\\\\toprule\n"
        output += "Temp & l & Samples \\\\ \\toprule\n"
        for temp in temps:
            stdf = sdf[sdf.temp == temp]
            output += s_multirow.format(4, temp) + "\n"
            for length in lengths:
                stldf = stdf[stdf.length == length]
                text = stldf.iloc[0].text
                if length == 50:
                    text = "\\\\".join([text[0:40], text[40:]])
                if length == 120:
                    text = "\\\\".join([text[0:40], text[40:80], text[80:]])
                output += s_text.format(length, text)
            output += "\\midrule\n"
        output += "\\end{tabularx}\n\\end{minipage}\n"
        # END TABLE
    output += "}" # CLOSE LAST FULL BLOCK
    with open(fpath + '.tex', 'w') as fp:
        fp.write(output)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str)
    args = parser.parse_args()
    steps = [0, 10000, 20000, 30000, 40000, 50000, 100000, 200000, 500000]
    make_table(args.file, steps)
