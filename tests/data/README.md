
# Provenance of the data


1. 230711_idleflow_400-1000mz_25mz_diaPasef_10sec.d
    - 10 seconds of diaPASEF of idle flow data collected using a timstofSCP.
2. DDPASEF_10seconds.d
    - 10 seconds of ddaPASEF of idle flow data collected using a timstof.
3. DDPASEF_10seconds.mzml
    - docker run --rm -it -v $PWD:/data mfreitas/tdf2mzml tdf2mzml.py -i /data/DDPASEF_10seconds.d
4. 230711_idleflow_400-1000mz_25mz_diaPasef_10sec.mzml
    -  docker run --rm -it -v $PWD:/data mfreitas/tdf2mzml tdf2mzml.py -i /data/230711_idleflow_400-1000mz_25mz_diaPasef_10sec.d