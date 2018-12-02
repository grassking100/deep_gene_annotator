gff_file="$1"
gff_file="${gff_file%.*}"
awk -F'\t' -v OFS="\t"  '
                         {   
                             
                             if(substr($1,1,1)=="#")
                             {
                                  #print($1)
                             }
                             else
                             {
                                new_chr=substr($1,4)
                                if(new_chr=="M")
                                { 
                                    #new_chr="mitochondria"
                                }
                                else
                                {
                                    if(new_char=="C")
                                    {
                                       #new_chr="chloroplast"
                                       print(new_chr,$2,$3,$4,$5,$6,$7,$8,$9)
                                    }
                                }
		                     }
                         }'  "${gff_file}.gff" > ${gff_file}_rename_chr.gff
                         
                         
