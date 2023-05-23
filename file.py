import csv
col_names=["class_index","class","height ","width","scale by ","what is","symtoms ","medicine"]
with open("class.csv", 'a+') as csvFile1:
                
    writer = csv.writer(csvFile1)
    writer.writerow(col_names)
    # writer.writerow(prods)
    csvFile1.close()