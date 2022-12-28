//
//  SkitagLSTM.swift
//  LSTM
//
//  Created by Miguel Santos Luparelli Mathieu on 6/11/22.
//

import GameplayKit

class lstmDATApreprocess {
    struct RandomNumberGeneratorWithSeed: RandomNumberGenerator {
        init(seed: Int) {
            // Set the random seed
            srand48(seed)
        }
        
        func next() -> UInt64 {
            // drand48() returns a Double, transform to UInt64
            return withUnsafeBytes(of: drand48()) { bytes in
                bytes.load(as: UInt64.self)
            }
        }
    }
    func train_eval_lstm_data(lstm_data:dataLSTM, lstm_labels:[Double], eventsWindow:Int) -> dataLSTMtrain_eval {
        
        var lstm_train_data:[[[Double]]] = []
        var lstm_train_label:[Double] = []
        var lstm_eval_data:[[[Double]]] = []
        var lstm_eval_label:[Double] = []
        
        // Features
        let feature_names:[String] = lstm_data.features ?? []
        //let gt_label:[Double] = lstm_data.label ?? []
        let gt_label:[Double] = lstm_labels
        var gt_label_slice:[Double] = []
        var feature_slice:[Double] = []
        
        // classes
        var train_count = [0, 0]
        var eval_count = [0, 0]
        
        // Split
        var generator = RandomNumberGeneratorWithSeed(seed:948) // 0.7364620938628159
        var count_training:Double = 0
        var count_seq_events:Double = 0
        var isTraining = true
        for indx in 0...(lstm_data.label?.count ?? 1)-eventsWindow{
            let isTraining1 = Bool.random(using:&generator)
            let isTraining2 = Bool.random(using:&generator)
            count_seq_events += 1
            if isTraining1 == true || isTraining2 == true {
                count_training += 1
                isTraining = true
            } else {
                isTraining = false
            }
            
            let indx_stop = indx + eventsWindow
            
            // label
            var event_label:Double = 0.0
            gt_label_slice = gt_label[indx..<indx_stop].map {Double($0)}
            let gt_label_ones = gt_label_slice.reduce(0,+)
            if gt_label_ones >= 4{
                event_label = 1
            }
            
            // features
            var feature_matrix:[[Double]] = []
            for feat_indx in 0...feature_names.count-1{
                let feature:[Double] = lstm_data.features_value_norm?[feat_indx] ?? []
                feature_slice = feature[indx..<indx_stop].map {Double($0)}
                feature_matrix.append(feature_slice)
            }
            let sequential_event = skitagLSTM.getTranspose(Wmatrix: feature_matrix)
            
            
            // balanced input data
            if isTraining{
                train_count[Int(event_label)] += 1
                lstm_train_data.append(sequential_event)
                lstm_train_label.append(event_label)
            } else {
                eval_count[Int(event_label)] += 1
                lstm_eval_data.append(sequential_event)
                lstm_eval_label.append(event_label)
            }
        }
        let lstm_data_train_eval = dataLSTMtrain_eval(train_values: lstm_train_data,
                                                      train_labels: lstm_train_label,
                                                      eval_values: lstm_eval_data,
                                                      eval_labels: lstm_eval_label)
        
        return lstm_data_train_eval
    }
    func count_class(labels:[Double]) -> [Int] {
        let ones = labels.reduce(0, +)
        let zeros = Double(labels.count) - ones
        let zero_ones = [Int(zeros), Int(ones)]
        return zero_ones
    }
    func balance_classes(lstm_data_train_eval: dataLSTMtrain_eval) -> dataLSTMtrain_eval{
        
        var lstm_train_data:[[[Double]]] = []
        var lstm_train_label:[Double] = []
        var lstm_eval_data:[[[Double]]] = []
        var lstm_eval_label:[Double] = []
        
        // train
        let train_zeros_ones = self.count_class(labels:lstm_data_train_eval.train_labels ?? [])
        let train_min_class:Int = train_zeros_ones.min() ?? 0
        var train_count = [0, 0]
        for indx in 0...(lstm_data_train_eval.train_labels?.count ?? 0)-1{
            let event_label:Int = Int(lstm_data_train_eval.train_labels?[indx] ?? 0)
            train_count[Int(event_label)] += 1
            if train_count[Int(event_label)] <= train_min_class{
                lstm_train_data.append(lstm_data_train_eval.train_values?[indx] ?? [])
                lstm_train_label.append(Double(event_label))
            }
        }
        
        //eval
        let eval_zeros_ones = self.count_class(labels:lstm_data_train_eval.eval_labels ?? [])
        let eval_min_class:Int = eval_zeros_ones.min() ?? 0
        var eval_count = [0, 0]
        for indx in 0...(lstm_data_train_eval.eval_labels?.count ?? 0)-1{
            let event_label:Int = Int(lstm_data_train_eval.eval_labels?[indx] ?? 0)
            eval_count[Int(event_label)] += 1
            if eval_count[Int(event_label)] <= eval_min_class{
                lstm_eval_data.append(lstm_data_train_eval.eval_values?[indx] ?? [])
                lstm_eval_label.append(Double(event_label))
            }
        }
        
        
        let lstm_data_train_eval_bal = dataLSTMtrain_eval(train_values: lstm_train_data,
                                                          train_labels: lstm_train_label,
                                                          eval_values: lstm_eval_data,
                                                          eval_labels: lstm_eval_label)
        
        return lstm_data_train_eval_bal
    }
}

class SkitagGaussianDistribution {
    private let randomSource: GKRandomSource
    let mean: Float
    let deviation: Float

    init(randomSource: GKRandomSource, mean: Float, deviation: Float) {
        precondition(deviation >= 0)
        self.randomSource = randomSource
        self.mean = mean
        self.deviation = deviation
    }

    func nextUniform() -> Float {
        guard deviation > 0 else { return mean }

        let x1 = randomSource.nextUniform() // a random number between 0 and 1
        let x2 = randomSource.nextUniform() // a random number between 0 and 1
        let z1 = sqrt(-2 * log(x1)) * cos(2 * Float.pi * x2) // z1 is normally distributed

        // Convert z1 from the Standard Normal Distribution to our Normal Distribution
        return z1 * deviation + mean
    }
}

class skitagLSTM {
    
    // Weights init
    class func initWeights(input:Int, hidden:Int, output:Int, seedrnd:Int) -> lstmWW{
        
        // architecture
        // lstm arch ["feature_8,mean,max,min", "8_18_2_LSTM_Wi", "8_18_2_LSTM_Wf", "8_18_2_LSTM_Wo", "8_18_2_LSTM_Wg", "19_2_LSTM_Why"]
        
        let arch_features = "feature_"+String(input)+",mean,max,min"
        let arch_wi = String(input)+"_"+String(hidden)+"_"+String(input+hidden+1)+"_LSTM_Wi"
        let arch_wf = String(input)+"_"+String(hidden)+"_"+String(input+hidden+1)+"_LSTM_Wf"
        let arch_wo = String(input)+"_"+String(hidden)+"_"+String(input+hidden+1)+"_LSTM_Wo"
        let arch_wg = String(input)+"_"+String(hidden)+"_"+String(input+hidden+1)+"_LSTM_Wg"
        let arch_why = String(hidden+1)+"_"+String(output)+"_LSTM_Why"
        let architecture = [arch_features, arch_wi, arch_wf, arch_wo, arch_wg, arch_why]
        
        // Wi
        let Wi = skitagLSTM.init_Weights_Matrix(rows:hidden, cols:hidden+input+1, seed:"skitag-Wi")
        let Wf = skitagLSTM.init_Weights_Matrix(rows:hidden, cols:hidden+input+1, seed:"skitag-Wf")
        let Wo = skitagLSTM.init_Weights_Matrix(rows:hidden, cols:hidden+input+1, seed:"skitag-Wo")
        let Wg = skitagLSTM.init_Weights_Matrix(rows:hidden, cols:hidden+input+1, seed:"skitag-Wg")
        let Why = skitagLSTM.init_Weights_Matrix(rows:output, cols:hidden+1, seed:"skitag-Why")
        
        
        let lstmWWeights = lstmWW(arch:architecture,
                                  WiStack:Wi,
                                  WfStack:Wf,
                                  WoStack:Wo,
                                  WgStack:Wg,
                                  Why:Why)
        
        return lstmWWeights
        
    }
    class func init_Weights_Matrix(rows:Int, cols:Int, seed:String) -> [[Double]]{
        
        // random values
        let weights_count = rows * cols
        let weights_seed = seed.data(using: .utf8)!
        let random = GKARC4RandomSource(seed:weights_seed)
        let weights_random = SkitagGaussianDistribution(randomSource: random, mean: 0, deviation: 1)
        
        // kaiming init weights. Why dimension is different fromw the rest.
        var kaiming:Double = sqrt(2.0/Double(cols))
        let isWhy = seed.contains("Why")
        if isWhy {
            kaiming = sqrt(2.0/Double(rows))
        }
        let weights_array = (0..<weights_count).map { _ in Double(weights_random.nextUniform()) }.map { $0 * kaiming}
        // let weights_stats = skitagSTATS().get_stats(values:weights_array)
        // print("mean", weights_stats.mean)
        // print("std", weights_stats.std)
        var weights_matrix:[[Double]] = []
        var indx_start = 0
        var indx_stop = indx_start + cols
        for _ in 0...rows-1{
            var weights_row = weights_array[indx_start..<indx_stop].map {Double($0)}
            
            // bias
            weights_row[0] = 0
            
            // append rows
            weights_matrix.append(weights_row)
            
            // update offsets
            indx_start += cols
            indx_stop += cols
        }
        //print("matrix dim", seed, weights_matrix.count, weights_matrix[0].count)
        
        return weights_matrix
    }
    class func init_Thetas(lstm_weights: lstmWW) -> lstmWW {
        
        let Wi:[[Double]] = lstm_weights.WiStack ?? []
        let Wf:[[Double]] = lstm_weights.WfStack ?? []
        let Wo:[[Double]] = lstm_weights.WoStack ?? []
        let Wg:[[Double]] = lstm_weights.WgStack ?? []
        let Why:[[Double]] = lstm_weights.Why ?? []
        
        // init thetas
        let dWi = Array(repeating: Array(repeating: 0.0, count: Wi[0].count), count: Wi.count)
        let dWf = Array(repeating: Array(repeating: 0.0, count: Wf[0].count), count: Wf.count)
        let dWo = Array(repeating: Array(repeating: 0.0, count: Wo[0].count), count: Wo.count)
        let dWg = Array(repeating: Array(repeating: 0.0, count: Wg[0].count), count: Wg.count)
        let dWhy = Array(repeating: Array(repeating: 0.0, count: Why[0].count), count: Why.count)
        
        let lstmWWthetas = lstmWW(WiStack:dWi,
                                  WfStack:dWf,
                                  WoStack:dWo,
                                  WgStack:dWg,
                                  Why:dWhy)
        return lstmWWthetas
    }
    
    //LSTMfeedprop
    class func getSEQprediction(skieventN:[[Double]], hsdim:Int, Wi:[[Double]], Wf:[[Double]], Wo:[[Double]], Wg:[[Double]], Why:[[Double]]) -> [Double]{
        
        //LSTM architecture
        LSTM.HS.removeAll()
        let initialHS:[Double] = Array(repeating: 0.0, count: hsdim)
        LSTM.HS.append(initialHS)
        LSTM.C.removeAll()
        let initialC:[Double] = Array(repeating: 0.0, count: hsdim)
        LSTM.C.append(initialC)
        LSTM.I.removeAll()
        LSTM.F.removeAll()
        LSTM.O.removeAll()
        LSTM.G.removeAll()
        //var Y:[[Double]] = []
        
        //sequential loop
        for s in 0...skieventN.count-1 {
            let LSTMoutput:[[Double]] = LSTMcellfeedprop(skievent:skieventN[s], hs:LSTM.HS[s], cell:LSTM.C[s], Wi:Wi, Wf:Wf, Wo:Wo, Wg:Wg, Why:Why)
            LSTM.C.append(LSTMoutput[0]);
            LSTM.HS.append(LSTMoutput[1]);
            LSTM.I.append(LSTMoutput[2]);
            LSTM.F.append(LSTMoutput[3]);
            LSTM.O.append(LSTMoutput[4]);
            LSTM.G.append(LSTMoutput[5]);
        }
        
        //output layer [bias,last hidden state]
        let biasHsO = [1]+LSTM.HS[LSTM.HS.count-1]
        let outputlayer = MatrixVectorMultiplication(wMatrix:Why, xArray:biasHsO)
        return outputlayer
    }
    class func LSTMcellfeedprop (skievent:[Double], hs:[Double], cell:[Double], Wi:[[Double]], Wf:[[Double]], Wo:[[Double]], Wg:[[Double]], Why:[[Double]]) -> [[Double]]{
        
        let biasHsX = [1]+hs+skievent
        
        
        
        //LSTM feedpropagation algorithm
        //input gate
        var f_inputgate:[Double] = []
        let inputgate = MatrixVectorMultiplication(wMatrix:Wi, xArray:biasHsX)
        inputgate.forEach{xoz in f_inputgate.append((1 / (1 + exp( -xoz))))}
        
        
        //forget gate
        var f_forgetgate:[Double] = []
        let forgetgate = MatrixVectorMultiplication(wMatrix:Wf, xArray:biasHsX)
        forgetgate.forEach{xoz in f_forgetgate.append((1 / (1 + exp( -xoz))))}
        
        
        //output gate
        var f_outputgate:[Double] = []
        let outputgate = MatrixVectorMultiplication(wMatrix:Wo, xArray:biasHsX)
        outputgate.forEach{xoz in f_outputgate.append((1 / (1 + exp( -xoz))))}
        
        
        //candidate cell
        var f_cellgate:[Double] = []
        let cellgate = MatrixVectorMultiplication(wMatrix:Wg, xArray:biasHsX)
        cellgate.forEach{xoz in f_cellgate.append( ((exp(xoz)-exp(-xoz)) / (exp(xoz)+exp(-xoz))) )}
        
        //cell value
        let cell_forget = elementWiseMultiplication(xArray1: cell, xArray2: f_forgetgate)
        let cell_input = elementWiseMultiplication(xArray1: f_cellgate, xArray2: f_inputgate)
        let cell_ = ArraySum(xArray1:cell_forget, xArray2:cell_input) //updated cell value
        
        //hidden state value
        var f_cell:[Double] = []
        cell_.forEach{xoz in f_cell.append( ((exp(xoz)-exp(-xoz)) / (exp(xoz)+exp(-xoz))) )} //xoz -> ((exp(xoz)-exp(-xoz)) / (exp(xoz)+exp(-xoz)))
        let hs_ = elementWiseMultiplication(xArray1:f_cell, xArray2:f_outputgate)
        
        let cellOutput:[[Double]] = [cell_,hs_,f_inputgate,f_forgetgate,f_outputgate,f_cellgate]
        
        return cellOutput
    }
    
    
    //
    // Operators
    //
    class func MatrixVectorMultiplication (wMatrix:[[Double]], xArray:[Double]) -> [Double]{
        var outputArray:[Double] = []
        for row in 0...(wMatrix.count-1){
            var rowSum:Double = 0.0
            for col in 0...(wMatrix[0].count-1){
                rowSum += (wMatrix[row][col]*xArray[col])
            }
            outputArray.append(rowSum)
        }
        return outputArray
    }
    class func matrixAccumulator(wMatrix1:[[Double]], wMatrix2:[[Double]]) -> [[Double]]{
        
        var matrixAccumulator = wMatrix1
        for row in 0...(wMatrix1.count-1){
            for col in 0...(wMatrix1[0].count-1){
                matrixAccumulator[row][col] += wMatrix2[row][col]
            }
        }
        return matrixAccumulator
    }
    class func elementWiseMultiplication (xArray1:[Double], xArray2:[Double]) -> [Double]{
        
        var xArray:[Double] = []
        for e in 0...(xArray1.count-1){
            xArray.append(xArray1[e]*xArray2[e])
        }
        return xArray
    }
    class func ArraySum(xArray1:[Double], xArray2:[Double]) -> [Double]{
        var xArray:[Double] = []
        for e in 0...(xArray1.count-1){
            xArray.append(xArray1[e]+xArray2[e])
        }
        
        return xArray
    }
    class func getSoftMax (outputlayer:[Double]) -> [Double] {
        
        var exp_outputlayer:[Double] = []
        outputlayer.forEach{o in exp_outputlayer.append( exp(o) )}
        let Eexp_output = exp_outputlayer.reduce(0) { $0 + $1 }
        var sm_outputlayer:[Double] = []
        exp_outputlayer.forEach{smo in sm_outputlayer.append( smo/Eexp_output )}
        
        return sm_outputlayer
    }
    class func softmaxClassSEQpred (aout:[Double]) -> [Double] {
        var aout_this = aout
        var maxvalue:Double = 1.0
        var maxvalue_next:Double = 0.0
        if aout_this.count > 0 {
            aout_this.sort(by: >)
            maxvalue = aout_this[0]
            maxvalue_next = aout_this[1]
        }
        let smClass:Double = Double(aout.firstIndex(of: maxvalue)!) //SIGTRAP exception; nil value
        let smClass_next:Double = Double(aout.firstIndex(of: maxvalue_next)!) //SIGTRAP exception; nil value
        let smClassPred:[Double] = [smClass, maxvalue, smClass_next]
        return smClassPred
    }
    class func getdW(row:[Double], col:[Double]) -> [[Double]]{
        
        var dW = Array(repeating: Array(repeating: 0.0, count: col.count), count: row.count)
        
        for i in 0...row.count-1{
            for j in 0...col.count-1{
                dW[i][j] = row[i] * col[j]
            }
        }
        return dW
    }
    
    
    
    
    
    //LSTM architecture
    class func getzsarray(skitagParamsL:[String], indxstart:Int, indxstop:Int) -> [[Double]]{
        
        var zsparams:[[Double]] = []
        var zsarray:[Double] = []
        for z in indxstart...indxstop-1{
            let zs = skitagParamsL[z].split(separator:",")
            for i in 1...zs.count-1{
                zsarray.append(Double(zs[i])!)
            }
            zsparams.append(zsarray)
            zsarray.removeAll()
        }
        return zsparams
    }
    class func getArchitecture(skitagParamsL:[String], indx:Int) -> [Int] {
        var skitagArch:[Int] = []
        let architecture = skitagParamsL[indx].split(separator:"_")
        for a in 0...architecture.count-2{
            skitagArch.append(Int(architecture[a])!)
        }
        return skitagArch
    }
    class func getArchitectureRNN(skitagParamsL:[String], indx:Int, Why:Bool) -> [Int] {
        var skitagArch:[Int] = []
        let architecture = skitagParamsL[indx].split(separator:"_")
        if (Why){
            for a in 0...architecture.count-3{
                skitagArch.append(Int(architecture[a])!)
            }
            skitagArch.append(skitagArch[skitagArch.count-1])
            skitagArch[skitagArch.count-2] = skitagArch[skitagArch.count-3]
        } else {
            for a in 0...architecture.count-3{
                skitagArch.append(Int(architecture[a])!)
            }
            skitagArch[skitagArch.count-1] = skitagArch[skitagArch.count-2]
        }
        return skitagArch
    }
    class func getLSTMweigths_depth(architecture:String) -> Int {
        let architecture = architecture.split(separator:"_")
        let weights_row_i = Int(architecture[0]) ?? 0
        let weights_row_h = Int(architecture[1]) ?? 0
        return weights_row_i + weights_row_h + 1
    }
    class func getArchitectureLength(architecture:[Int]) -> Int {
        var architecturelength:Int = 0
        for a in 0...architecture.count-2{
            architecturelength = architecturelength + Int((architecture[a]+1) * architecture[a+1])
        }
        return architecturelength
    }
    class func getArchitectureLengthRNN(architecture:[Int]) -> Int {
        var architecturelength:Int = 0
        for a in 0...architecture.count-2{
            if (a == 0){
                architecturelength = architecturelength + Int((architecture[a]) * architecture[a+1])
            } else {
                architecturelength = architecturelength + Int((architecture[a]+1) * architecture[a+1])
            }
        }
        return architecturelength
    }
    class func weigthsArray(indxstart:Int, indxstop:Int, skitagParamsL:[String], architecturelength:Int) -> [Double]{
        var Warray:[Double] = []
        for w in 0...architecturelength-1{
            Warray.append(Double(skitagParamsL[w+indxstart])!);
        }
        return Warray
    }
    class func stringTOarray(string_array:String) -> [Double]{
        
        var double_array:[Double] = []
        let string_array_list = string_array.split(separator: ",")
        for val in 0...string_array_list.count - 1{
            double_array.append(Double(string_array_list[val])!)
        }
        return double_array
    }
    class func stackWeightsArrays(indxstart:Int, indxstop:Int, skitagParamsL:[String]) -> [[Double]]{
        var weights:[[Double]] = []
        for indx in indxstart...indxstop{
            weights.append(stringTOarray(string_array:skitagParamsL[indx]))
        }
        return weights
    }
    
    
    //get thetas
    class func getGatesList (thetasArray:[Double], thetasArchitecture:[Int]) -> [[[Double]]] {
        
        var GatesList:[[[Double]]] = []
        
        var startindx = 0
        var stopindx = 0
        var row = 0
        var col = 0
        for layer in 1 ... thetasArchitecture.count-1 {
            if (layer > 1){
                row = thetasArchitecture[layer-1]+1
            } else if (layer == 1){
                row = thetasArchitecture[layer-1]
            }
            col = thetasArchitecture[layer]
            stopindx = startindx + (row*col)
            let thArray = Array(thetasArray[startindx...stopindx-1])
            startindx = stopindx
            var thetasMatrix = Array(repeating: Array(repeating: 0.0, count: col), count: row)
            var ij:Int = 0
            for i in 0 ... (row-1) {
                for j in 0 ... (col-1) {
                    thetasMatrix[i][j] = thArray[ij]
                    ij += 1
                }
            }
            GatesList.append(thetasMatrix)
        }
        return GatesList
    }
    
    class func getTranspose (Wmatrix:[[Double]]) -> [[Double]] {
        let row = Wmatrix.count
        let col = Wmatrix[0].count
        var thetasMatrixT = Array(repeating: Array(repeating: 0.0, count: row), count: col)
        for j in 0...(col-1){
            for i in 0...(row-1){
                thetasMatrixT[j][i] = Wmatrix[i][j];
            }
            
        }
        return thetasMatrixT
    }
    
    class func getModeArray(skitag_pred:[Double]) -> Double{
        let skitag_mode = Dictionary(grouping: skitag_pred, by: { $0 }).map( { [Int($0.key), $0.value.count] } )
        let skitag_key = skitag_mode.map( {$0[0]} )
        let skitag_value = skitag_mode.map( {$0[1]} )
        let skitag_value_max = skitag_value.max() ?? 0
        let skitag_value_indx = skitag_value.firstIndex(of: skitag_value_max) ?? 0
        let skitag_pred_mode:Double = Double(skitag_key[skitag_value_indx])
        return skitag_pred_mode
    }
    
    
    
    
    
    //
    // Weights setup
    //
    func getLSTMweigthsWW(paramsL:[String]) -> lstmWW {
        
        var indxparams_start = 0;
        var indxparams_stop = indxparams_start;
        var architecture:String = ""
        var rows:Int = 0
        var arch:[String] = []
        var lstmW = lstmWW()
        
        //zs params
        architecture = paramsL[indxparams_start]
        arch.append(architecture)
        let imuzsfeaturesSeq = paramsL[indxparams_start].split(separator:",")[0]
        let imuzsfeaturesSeq_ = Int(String(imuzsfeaturesSeq.split(separator:"_")[1]))!
        indxparams_start+=1;
        indxparams_stop = indxparams_start+imuzsfeaturesSeq_;
        lstmW.zsSeq = skitagLSTM.getzsarray(skitagParamsL:paramsL, indxstart:indxparams_start, indxstop:indxparams_stop)
        
        //input gate
        indxparams_start = indxparams_stop
        architecture = paramsL[indxparams_start]
        arch.append(architecture)
        rows = skitagLSTM.getLSTMweigths_depth(architecture:architecture)
        indxparams_start+=1
        indxparams_stop = indxparams_start + rows - 1;
        let skitagDOWNHILLthetasWi_stack = skitagLSTM.stackWeightsArrays(indxstart:indxparams_start, indxstop:indxparams_stop, skitagParamsL:paramsL)
        // lstmW.hsdim = skitagDOWNHILLthetasWi_stack[0].count
        lstmW.WiStack = skitagLSTM.getTranspose(Wmatrix: skitagDOWNHILLthetasWi_stack) //transpose Wi matrix
        

        //forget gate
        indxparams_start = indxparams_stop+1
        architecture = paramsL[indxparams_start]
        arch.append(architecture)
        rows = skitagLSTM.getLSTMweigths_depth(architecture:architecture)
        indxparams_start+=1
        indxparams_stop = indxparams_start + rows - 1;
        let skitagDOWNHILLthetasWf_stack = skitagLSTM.stackWeightsArrays(indxstart:indxparams_start, indxstop:indxparams_stop, skitagParamsL:paramsL)
        lstmW.WfStack = skitagLSTM.getTranspose(Wmatrix: skitagDOWNHILLthetasWf_stack) //transpose Wi matrix

        //output gate
        indxparams_start = indxparams_stop+1
        architecture = paramsL[indxparams_start]
        arch.append(architecture)
        rows = skitagLSTM.getLSTMweigths_depth(architecture:architecture)
        indxparams_start+=1
        indxparams_stop = indxparams_start + rows - 1;
        let skitagDOWNHILLthetasWo_stack = skitagLSTM.stackWeightsArrays(indxstart:indxparams_start, indxstop:indxparams_stop, skitagParamsL:paramsL)
        lstmW.WoStack = skitagLSTM.getTranspose(Wmatrix: skitagDOWNHILLthetasWo_stack) //transpose Wi matrix

        //cell gate
        indxparams_start = indxparams_stop+1
        architecture = paramsL[indxparams_start]
        arch.append(architecture)
        rows = skitagLSTM.getLSTMweigths_depth(architecture:architecture)
        indxparams_start+=1
        indxparams_stop = indxparams_start + rows - 1;
        let skitagDOWNHILLthetasWg_stack = skitagLSTM.stackWeightsArrays(indxstart:indxparams_start, indxstop:indxparams_stop, skitagParamsL:paramsL)
        lstmW.WgStack = skitagLSTM.getTranspose(Wmatrix: skitagDOWNHILLthetasWg_stack) //transpose Wi matrix

        //why
        indxparams_start = indxparams_stop+1
        architecture = paramsL[indxparams_start]
        arch.append(architecture)
        rows = Int(architecture.split(separator:"_")[0]) ?? 0
        indxparams_start+=1
        indxparams_stop = indxparams_start + rows - 1;
        let skitagDOWNHILLthetasWhy_stack = skitagLSTM.stackWeightsArrays(indxstart:indxparams_start, indxstop:indxparams_stop, skitagParamsL:paramsL)
        lstmW.Why = skitagLSTM.getTranspose(Wmatrix: skitagDOWNHILLthetasWhy_stack) //transpose Wi matrix
        
        // architecture
        lstmW.arch = arch
        return lstmW
    }
    
    func lstmWeights_String(lstm_weights: lstmWW, features_names:[String]) -> String {
        
        var lstm_weights_str = ""
        
        //save zs values
        let lstm_arch:[String] = lstm_weights.arch ?? []
        let lstm_zsSeq:[[Double]] = lstm_weights.zsSeq ?? []
        lstm_weights_str += lstm_arch[0] + "\n"
        for row in 0...lstm_zsSeq.count - 1{
            let row_string = String(features_names[row])+","+"\(lstm_zsSeq[row])".replacingOccurrences(of: " ", with: "").replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "")+"\n"
            lstm_weights_str += row_string
        }
        
        //saveWi
        let WiStack:[[Double]] = lstm_weights.WiStack ?? []
        let WiT = skitagLSTM.getTranspose(Wmatrix: WiStack)
        lstm_weights_str += lstm_arch[1] + "\n"
        for row in 0...WiT.count - 1{
            let row_string = "\(WiT[row])".replacingOccurrences(of: " ", with: "").replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "")+"\n"
            lstm_weights_str += row_string
        }
        
        //saveWf
        let WfStack:[[Double]] = lstm_weights.WfStack ?? []
        let WfT = skitagLSTM.getTranspose(Wmatrix: WfStack)
        lstm_weights_str += lstm_arch[2] + "\n"
        for row in 0...WfT.count - 1{
            let row_string = "\(WfT[row])".replacingOccurrences(of: " ", with: "").replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "")+"\n"
            lstm_weights_str += row_string
        }
        
        //saveWo
        let WoStack:[[Double]] = lstm_weights.WoStack ?? []
        let WoT = skitagLSTM.getTranspose(Wmatrix: WoStack)
        lstm_weights_str += lstm_arch[3] + "\n"
        for row in 0...WoT.count - 1{
            let row_string = "\(WoT[row])".replacingOccurrences(of: " ", with: "").replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "")+"\n"
            lstm_weights_str += row_string
        }
        
        //saveWg
        let WgStack:[[Double]] = lstm_weights.WgStack ?? []
        let WgT = skitagLSTM.getTranspose(Wmatrix: WgStack)
        lstm_weights_str += lstm_arch[4] + "\n"
        for row in 0...WgT.count - 1{
            let row_string = "\(WgT[row])".replacingOccurrences(of: " ", with: "").replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "")+"\n"
            lstm_weights_str += row_string
        }
        
        //saveWhy
        let WhyStack:[[Double]] = lstm_weights.Why ?? []
        let WhyT = skitagLSTM.getTranspose(Wmatrix: WhyStack)
        lstm_weights_str += lstm_arch[5] + "\n"
        for row in 0...WhyT.count - 1{
            let row_string = "\(WhyT[row])".replacingOccurrences(of: " ", with: "").replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "")+"\n"
            lstm_weights_str += row_string
        }
        // print(lstm_weights_str)
        return lstm_weights_str
    }
}
