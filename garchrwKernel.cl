__kernel void garchrw(
                    __global float* output, //output vector
                    const unsigned int nSim, //num of simuls
                    const unsigned int nPer, //num of forecasting pers
                    const __global float* rands, //vector of random unif (0,1) vals
                    const float mu, //RW drift term
                    const unsigned int gOrder, //garch g order
                    const unsigned int arOrder, //garch ar order
                    const float meanCoef, //garch mean
                    const __global float* GCoefs, //vector of garch g coefs
                    const __global float* ARCoefs, //vector of garch ar coefs
                    const __global float* sample, //in-sample data vector
                    const unsigned int sampleLength)
//routine
{
    //number of coefs
    uint gCoefsSize = sizeof(GCoefs) / sizeof(*GCoefs);
    uint ARCoefsSize = sizeof(ARCoefs) / sizeof(*ARCoefs);
    
    //conditional means and variances
    float condMeans[sampleLength + nPer];
    float condVar[sampleLength + nPer];
    
    //fill conditionals with available data from sample
    for(int i = 0; i < sampleLength; i++){
        condMeans[i] = sample[i];
        condVar[i] = (sample[i] - mu)*(sample[i] - mu);
    }
    //here goes GARCH correction
    size_t gid = get_global_id(0);
    if(gid < nSim){
        for(int per = 0; per < nPer; per++){
            //mean
            float var = meanCoef;
            //p returns
            for(int p = 0; p < gCoefsSize; p++){
                var += GCoefs[p] * condMeans[sampleLength - 1 - p + per] * condMeans[sampleLength - 1 - p + per];
            }
            for(int q = 0; q < ARCoefsSize; q++){
                var += ARCoefs[q] * condVar[sampleLength - 1 - q + per];
            }
            condVar[sampleLength + per] = var;
            condMeans[sampleLength + per] = rands[gid*nPer + per]*sqrt(var) + mu;
            output[gid*nPer + per] = condMeans[sampleLength + per];
        }
        //printf("%s/n", gid);
    }
};