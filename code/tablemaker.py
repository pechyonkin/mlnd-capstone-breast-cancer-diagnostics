def write_to_a_table(df, 
                     path_to_output, 
                     scap = 'Test cap',
                     caption = 'Test caption', 
                     label = 'lest-label', 
                     first_column_header = 'Column',
                     tabname = 'test-tab', 
                     precision = 2, 
                     verbose=True):
    ''' this routine takes a DataFrame
        and outputs it as a nice .tex table
        to be put in my PDF Capstone Report
        ---
        df: original DataFrame
        path_to_output: where to put the table
        scap: Tex short caption
        caption: Tex caption
        label: Tex reference label
        first_column_header
        tabname: name of the file *.tex
        precision: number of decimal places
        ---
        sometimes it won't render correcly in .tex
        when there are .tex characters that need to be escaped
        so I need to change values in .tex file manually
        ---
        only tested on Mac OS X'''

    n_col = "c" + ("S[table-format=2." + str(precision) + ", round-precision=" + str(precision) + "]") * (df.shape[1])
    beginning_lines = [
        "\\begin{table}[htb]",
        "\\centering",
        "\\sisetup{output-decimal-marker = {.}}"
        "\\caption[" + scap + "]{" + caption + "}",
        "\\label{tab:" + label + "}",
        "\\begin{tabular}{" + n_col +"}",
        "\\toprule"
    ]
    
    # make headers
    head = []
    head.append("\\multicolumn{1}{c}{\\textbf{" + first_column_header + "}}")
    for e in df.columns:
        # replace percents with escape percents
        e = str(e).replace('%','\%')
        head.append("\\multicolumn{1}{c}{\\textbf{" + e + "}}")
    # head
    headers = " & ".join(head) + "\\"
    headers
    
    # make lines
    middle_lines = []
    for i in df.index:
        line = str(i)
        for e in df.ix[df.index.get_loc(i)]:
            line += " & " + ("{:." + str(precision) + "f}").format(e)
        line += " \\\\" + "\n"
        middle_lines.append(line)
    
    ending_lines = [
        "\\midrule",
        "\\end{tabular}",
        "\\end{table}"
    ]
     
    f = open(path_to_output + tabname + '.tex', 'w')
    # write the beginning of table
    for l in beginning_lines:
        f.write(l + '\n')
    # put the header line
    f.write(headers + '\\' + '\n')
    # put a line
    f.write("\\midrule\n")
    # put the body of table
    for l in middle_lines:
        f.write(l)
    # put the bottom of table
    for l in ending_lines:
        f.write(l + '\n')
    f.close()
    if verbose:
	    print "-"*80
	    print "SUCESSFUL OUTPUT TO A .TEX TABLE"
	    print "-"*80