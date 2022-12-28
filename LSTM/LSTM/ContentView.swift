//
//  ContentView.swift
//  LSTM
//
//  Created by Miguel Santos Luparelli Mathieu on 6/11/22.
//

import SwiftUI
import Charts

struct ContentView: View {
    
    // LSTM
    @StateObject private var tr = LSTMTrainManager()
    
    // LSTM params
    @State private var input_data = ""
    @State private var lstm_data = dataLSTM()
    @State private var lstm_weights = lstmWW()
    @State private var lstm_thetas = lstmWW()
    @State private var lstm_params = lstmPARAMS()
    @State private var seq_event_count = 0
    @State private var epochs = 100
    @State private var indx_w = 6
    @State private var hidden = 8
    @State private var Jout:Double = 0
    
    // Chart
    @State private var trainingdata:[dataChart] = []
    @State private var xaxis:[Double] = []
    @State private var isThinking = false
    
    
    
    var body: some View {
        VStack (alignment: .leading){
            VStack{
                Spacer()
                HStack{
                    Text("Skitag")
                        .font(.system(size: 22).weight(.bold))
                        .padding([.leading, .top, .bottom])
                    Text("AI System")
                        .font(.system(size: 16).weight(.light))
                        .padding([.top, .bottom])
                    Image(systemName: "brain")
                        .padding([.top, .bottom])
                    Spacer()
                }.padding([.bottom, .leading])
                Divider()
                    .frame(minWidth: 2)
                    .overlay(.white)
                    .padding([.bottom])
            }// ends header
            .frame(height:140)
            .background(Color("ColorAccent"))
            .foregroundColor(.white)
            
            GeometryReader{ geom in
                ZStack{
                    VStack{
                        HStack{
                            Text("Training Cost Chart").padding()
                            if self.isThinking == true {
                                SkitagActivity()
                            }
                            Spacer()
                        }
                        Chart (trainingdata) { dc in
                            LineMark(
                                x: .value("Epoch", dc.epoch),
                                y: .value("Cost", dc.costv)
                            )
                        }
                        Spacer()
                    }
                }
                .padding()
                .frame(width: geom.size.width, height: geom.size.height)
            }.background(.white)
            
            Spacer()
            HStack{
                Spacer()
                HStack{
                    HStack{
                        Image(systemName: "brain")
                    }
                }
                .frame(width: 60, height:60)
                .foregroundColor(.white)
                .background(Color("ColorAccent"))
                .cornerRadius(30)
                .padding([.trailing, .bottom], 50)
                .onTapGesture{
                    //print("Training... ")
                    self.isThinking.toggle()
                }
            }
            .fullScreenCover(isPresented: $isThinking, onDismiss: on_close){
                VStack{
                    HStack{
                        Text("Training a LSTM...")
                            .font(.system(size: 32, weight: .regular))
                            .padding()
                        SkitagActivity()
                    }
                    Divider()
                    ScrollView (.vertical){
                        VStack{
                            HStack{
                                Text("It takes 35 seconds (aprox) to complete this task. Do you think it is too much? Think about the months (even years) that takes to train an AI System of the GPT3 kind. BTW: By the time you finish reading this text, the task will be probably half completed. In the meantime, try a breath exercise... Breathe in through your nose,... Breathe out with pursed lips as if you were going to blow out a candle. Try to breathe out longer than your inhale,... Repeat a few times,... Stop if you feel light-headed. Thanks for reading! This screen will be automatically closed when the training is completed. Enjoy it!!! ...")
                                    .font(.system(size: 24, weight: .light))
                                
                            }
                        }
                    }.padding([.top, .bottom, .leading, .trailing], 30)
                }.onAppear(perform: trainLSTM)
                
                
            }
        }
        .onAppear(perform: on_appear)
        
        .ignoresSafeArea(edges:.top)
    }
    func on_appear(){
        self.input_data = self.read_csv()
    }
    func on_close(){
        
    }
    func trainLSTM(){
        self.lstm_data = self.parse_lstm_data_ondevice(lstm_data_string: self.input_data)
        tr.lstm_setup(lstm_data: self.lstm_data)
        self.trainingdata.removeAll()
        DispatchQueue.main.async{
            for epoch in 0..<100{
                
                //feed and back propagation
                tr.lstm_feedback_propagation(lstm_train_eval: tr.lstm_train_eval_bal)
                
                // update weights
                tr.lstm_update_weights()
                
                // data to plot
                // print(tr.Jout)
                self.trainingdata.append(dataChart(epoch:Double(epoch), costv: tr.Jout))
            }
        }
        self.isThinking.toggle()
    }
    func read_csv() -> String {
        
        var input_data = ""
        let data_csv = Bundle.main.path(forResource: "/Data", ofType: "csv")
        let data_csvURL:URL = URL(fileURLWithPath: data_csv!)
        let exist_input_data = FileManager.default.fileExists(atPath: data_csvURL.path)
        if !exist_input_data {
            return input_data
        }
        do {
            input_data = try String(contentsOf: data_csvURL)
        } catch let error as NSError {
            print ("Failed to read file", error)
            return input_data
        }
        return input_data
    }
    func parse_lstm_data_ondevice(lstm_data_string:String) -> dataLSTM{
        let lstm_data_string_ = lstm_data_string.dropLast()
        let lstm_data_string_list = lstm_data_string_.split(separator:";")
        
        var iot_name:String = ""
        var feature_name_list:[String] = []
        var gd_truth_label:[Double] = []
        var feature_norm_list:[dataNORM] = []
        var feature_values_matrix:[[Double]] = []
        
        for data_row in lstm_data_string_list{
            let data_row_key = data_row.split(separator:":")
            switch String(data_row_key[0]){
            case "iot":
                iot_name = String(data_row_key[1])
            case "features":
                feature_name_list = data_row_key[1].split(separator:",").map ({String($0)})
            case "label":
                gd_truth_label = data_row_key[1].split(separator:",").map ({Double($0) ?? 0})
            case "features_norms":
                let feature_norms = data_row_key[1].split(separator:"|")
                for feat in feature_norms{
                    
                    // feature elements
                    var norm_name = ""
                    var mean:Double = 0
                    var max:Double = 0
                    var min:Double = 0
                    
                    let feat_norms_elements = feat.split(separator:",")
                    for feat_element in feat_norms_elements{
                        let feat_key_value = feat_element.split(separator:"=")
                        let feat_key = String(feat_key_value[0])
                        
                        switch feat_key{
                        case "name":
                            norm_name = String(feat_key_value[1])
                        case "mean":
                            mean = Double(feat_key_value[1]) ?? 0
                        case "max":
                            max = Double(feat_key_value[1]) ?? 0
                        case "min":
                            min = Double(feat_key_value[1]) ?? 0
                        default:
                            continue
                        }
                    }
                    let norm_params = dataNORM(name: norm_name,
                                               mean: mean,
                                               max: max,
                                               min: min)
                    feature_norm_list.append(norm_params)
                }
            case "features_value":
                let feature_values_rows = data_row_key[1].split(separator:"|")
                for indx in 0...feature_values_rows.count-1{
                    let feature_values_row_double = feature_values_rows[indx].split(separator:",").map ({Double($0) ?? 0})
                    feature_values_matrix.append(feature_values_row_double)
                }
            default:
                continue
            }
        }
        let lstm_data_read = dataLSTM(iot: iot_name,
                                      features: feature_name_list,
                                      label: gd_truth_label,
                                      features_norms: feature_norm_list,
                                      features_value_raw: feature_values_matrix,
                                      features_value_norm: feature_values_matrix)
        
        return lstm_data_read
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

class LSTMTrainManager:ObservableObject{
    
    @State private var ds = lstmDATApreprocess()
    
    // LSTM
    var lstm_weights = lstmWW()
    var lstm_thetas = lstmWW()
    var lstm_params = lstmPARAMS()
    var lstm_train_eval_bal = dataLSTMtrain_eval()
    var seq_event_count = 0
    
    // Published variables
    @Published var epochs = 5
    @Published var indx_w = 6
    @Published var hidden = 8
    @Published var Jout:Double = 1
    
    func lstm_setup(lstm_data: dataLSTM){
        let input_size = lstm_data.features?.count ?? 0
        //print("input size", input_size, "events", lstm_data.label?.count ?? 0)
        
        // split data
        let lstm_train_eval = ds.train_eval_lstm_data(lstm_data:lstm_data, lstm_labels: lstm_data.label ?? [], eventsWindow: self.indx_w)
        
        self.lstm_train_eval_bal = ds.balance_classes(lstm_data_train_eval: lstm_train_eval)
        let class_train_bal = ds.count_class(labels:lstm_train_eval_bal.train_labels ?? [])
        let class_eval_bal = ds.count_class(labels:lstm_train_eval_bal.eval_labels ?? [])
        // print("train balanced", class_train_bal)
        // print("eval balanced", class_eval_bal)
        
        // init weights
        self.lstm_weights = skitagLSTM.initWeights(input:input_size, hidden:self.hidden, output:2, seedrnd:3456)
        // print("lstm architecture", self.lstm_weights.arch ?? [])
        // print("try to save the lstm_weights")
        // print(self.lstm_weights)

        // init thetas
        self.lstm_thetas = skitagLSTM.init_Thetas(lstm_weights: self.lstm_weights)
        // print("thetas dim", self.lstm_thetas.WiStack?.count ?? 0, self.lstm_thetas.WiStack?[0].count ?? 0)
    }
    
    func on_train(lstm_data: dataLSTM){
        
        let input_size = lstm_data.features?.count ?? 0
        print("input size", input_size, "events", lstm_data.label?.count ?? 0)
        
        // print(lstm_data)
        
        // split data
        let lstm_train_eval = ds.train_eval_lstm_data(lstm_data:lstm_data, lstm_labels: lstm_data.label ?? [], eventsWindow: self.indx_w)
        
        let lstm_train_eval_bal = ds.balance_classes(lstm_data_train_eval: lstm_train_eval)
        let class_train_bal = ds.count_class(labels:lstm_train_eval_bal.train_labels ?? [])
        let class_eval_bal = ds.count_class(labels:lstm_train_eval_bal.eval_labels ?? [])
        print("train balanced", class_train_bal)
        print("eval balanced", class_eval_bal)
        
        // init weights
        self.lstm_weights = skitagLSTM.initWeights(input:input_size, hidden:self.hidden, output:2, seedrnd:3456)
        print("lstm architecture", self.lstm_weights.arch ?? [])

        // init thetas
        self.lstm_thetas = skitagLSTM.init_Thetas(lstm_weights: self.lstm_weights)
        print("thetas dim", self.lstm_thetas.WiStack?.count ?? 0, self.lstm_thetas.WiStack?[0].count ?? 0)
        
        // sequential data
        for indx in 0..<self.epochs{
            
            // print(self.lstm_weights.WiStack)
            
            // feed back propagation
            self.lstm_feedback_propagation(lstm_train_eval: lstm_train_eval_bal)
            print("epoch", indx+1, self.Jout)
            
            // update weights
            self.lstm_update_weights()
        }
    }
    
    //
    // LSTM feedforward and backpropagation algorithm
    //
    func lstm_feedback_propagation(lstm_train_eval: dataLSTMtrain_eval) {
        
        // Sequential Events
        self.seq_event_count = lstm_train_eval.train_values?.count ?? 1
        // print("events", self.seq_event_count)
        
        // Weigths
        let hsdim = String(lstm_weights.arch?[1].split(separator: "_")[1] ?? "")
        let hsdim_int = Int(hsdim) ?? 0
        
        var Jout:Double = 0
        
        for indx in 0...self.seq_event_count-1 {
            
            // sequential event
            let event_label = lstm_train_eval.train_labels?[indx] ?? 0
            let sequential_event = lstm_train_eval.train_values?[indx] ?? []
            
            // prediction
            if sequential_event.count > 0 {
                let eventOUT = skitagLSTM.getSEQprediction(skieventN:sequential_event,
                                                           hsdim:hsdim_int,
                                                           Wi:self.lstm_weights.WiStack ?? [],
                                                           Wf:self.lstm_weights.WfStack ?? [],
                                                           Wo:self.lstm_weights.WoStack ?? [],
                                                           Wg:self.lstm_weights.WgStack ?? [],
                                                           Why:self.lstm_weights.Why ?? [])
                let eventPREDsoftmax = skitagLSTM.getSoftMax(outputlayer:eventOUT)
                
                //filter NaN values; SIGTRAP Thread solved
                var lstmPRED:[Double] = []
                let filternan = eventPREDsoftmax.max()!
                if filternan.isNaN == false{
                    lstmPRED = skitagLSTM.softmaxClassSEQpred(aout:eventPREDsoftmax)
                } else {
                    lstmPRED.append(9)
                }
                
                // loss function
                Jout += -log(eventPREDsoftmax[Int(event_label)])
                
                // get error from: eventPREDsoftmax
                var delta_error = eventPREDsoftmax
                delta_error[Int(event_label)] -= 1
                // print(event_label, lstmPRED[0], eventPREDsoftmax, delta_error, eventOUT)
                
                // Backpropagate error
                self.LSTMbackpropagation(delta_error:delta_error,
                                               sequential_event:sequential_event,
                                               hsdim:hsdim_int)
            }
        }
        self.Jout = Jout/Double(seq_event_count)
    }
    func LSTMbackpropagation(delta_error:[Double],
                                   sequential_event:[[Double]],
                                   hsdim:Int){
        
        
        //derivatives, deltas arrays
        var dhs = Array(repeating: 0.0, count: hsdim)
        var dhs_next = Array(repeating: 0.0, count: hsdim) //delta_hs<t+1>
        var dc_next = Array(repeating: 0.0, count: hsdim) //delta_c<t+1>
        
        let layers = sequential_event.count
        for RNNlayer in (0...layers-1).reversed(){
            
            //x<t>, hs<t> and c<t> layer index; this is OK
            let xl = RNNlayer; //x input; hidden state activation: i, f, o, g
            let hsl = RNNlayer+1; //hidden state
            let cl = RNNlayer+1; //cell gate
            
            //dypred
            var dypred:[Double] = []
            if (xl == layers-1){
                //Many to One architecture;dypred = delta_outList.get(delta_outList.size()-1);
                dypred = delta_error
            } else {
                dypred = Array(repeating: 0.0, count: delta_error.count)
                
            }
            
            //cell params; this is OK
            let hs_this = LSTM.HS[hsl]; //hs<t>
            let hs_prev = LSTM.HS[hsl-1]; //hs<t-1>
            
            let xl_this = sequential_event[xl]; //x<t> //normalise data before backpropagate data
            
            let c_this = LSTM.C[cl]; //c<t>
            let c_prev = LSTM.C[cl-1]; //c<t-1>
            
            //LSTM gates; this is OK
            let hsi = LSTM.I[xl]; //input gate
            let hsf = LSTM.F[xl]; //forget gate
            let hso = LSTM.O[xl]; //output gate
            let hsg = LSTM.G[xl]; //cell gate
            
            //dWhy
            var dWhy = skitagLSTM.getdW(row:hs_this, col:dypred)
            dWhy.insert(dypred, at: 0)
            let dWhyT = skitagLSTM.getTranspose(Wmatrix: dWhy)
            self.lstm_thetas.Why = skitagLSTM.matrixAccumulator(wMatrix1: self.lstm_thetas.Why ?? [], wMatrix2: dWhyT)
            
            //dhs last hidden state
            var WhyT = skitagLSTM.getTranspose(Wmatrix:lstm_weights.Why ?? []) //transpose Why
            WhyT.removeFirst() //remove bias
            dhs = skitagLSTM.MatrixVectorMultiplication(wMatrix:WhyT, xArray:dypred)
            dhs = skitagLSTM.ArraySum(xArray1:dhs, xArray2:dhs_next)
            
            
            //dhso, output gate derivative
            var dhso:[Double] = []
            c_this.forEach{xoz in dhso.append( ((exp(xoz)-exp(-xoz)) / (exp(xoz)+exp(-xoz))) )}
            dhso = skitagLSTM.elementWiseMultiplication(xArray1:dhso, xArray2:dhs)
            var dsigm_hso:[Double] = []
            hso.forEach{xoz in dsigm_hso.append( xoz * (1 - xoz) )}
            dhso = skitagLSTM.elementWiseMultiplication(xArray1:dsigm_hso, xArray2:dhso)
            
            
            //delta_c
            var dtanh_c:[Double] = []
            c_this.forEach{xoz in dtanh_c.append( 1 - pow(((exp(xoz)-exp(-xoz)) / (exp(xoz)+exp(-xoz))),2) )}
            var dc = skitagLSTM.elementWiseMultiplication(xArray1:dhs, xArray2:dtanh_c)
            dc = skitagLSTM.elementWiseMultiplication(xArray1:dc, xArray2:hso)
            dc = skitagLSTM.ArraySum(xArray1:dc, xArray2:dc_next)
            
            
            //dhsf
            var dhsf = skitagLSTM.elementWiseMultiplication(xArray1:c_prev, xArray2:dc)
            var dsigm_delta_hsf:[Double] = []
            hsf.forEach{xoz in dsigm_delta_hsf.append( xoz * (1 - xoz) )}
            dhsf = skitagLSTM.elementWiseMultiplication(xArray1:dsigm_delta_hsf, xArray2:dhsf)
            
            
            //dhsi
            var dhsi = skitagLSTM.elementWiseMultiplication(xArray1:hsg, xArray2:dc)
            var dsigm_delta_hsi:[Double] = []
            hsi.forEach{xoz in dsigm_delta_hsi.append( xoz * (1 - xoz) )}
            dhsi = skitagLSTM.elementWiseMultiplication(xArray1:dsigm_delta_hsi, xArray2:dhsi)
            
            
            //dhc
            var dhsg = skitagLSTM.elementWiseMultiplication(xArray1:hsi, xArray2:dc)
            var dsigm_delta_hsc:[Double] = []
            hsg.forEach{xoz in dsigm_delta_hsc.append( (1 - pow(xoz,2)) )}
            dhsg = skitagLSTM.elementWiseMultiplication(xArray1:dsigm_delta_hsc, xArray2:dhsg)
            
            //Accumulate derivatives
            
            //get derivatives, dWf
            var WfT = skitagLSTM.getTranspose(Wmatrix:self.lstm_weights.WfStack ?? []) //transpose Wf
            WfT.removeFirst() //remove bias
            let dXf = skitagLSTM.MatrixVectorMultiplication(wMatrix:WfT, xArray:dhsf)
            
            //dWff y dWfxh
            var dWfhh = skitagLSTM.getdW(row:hs_prev, col:dhsf)
            dWfhh.insert(dhsf, at: 0)
            let dWfxh = skitagLSTM.getdW(row:xl_this, col:dhsf)
            let dWf = dWfhh + dWfxh
            let dWfT = skitagLSTM.getTranspose(Wmatrix: dWf)
            self.lstm_thetas.WfStack = skitagLSTM.matrixAccumulator(wMatrix1: self.lstm_thetas.WfStack ?? [], wMatrix2: dWfT)
            
            
            //get derivatives, dWi
            var WiT = skitagLSTM.getTranspose(Wmatrix:self.lstm_weights.WiStack ?? []) //transpose Wi
            WiT.removeFirst() //remove bias
            let dXi = skitagLSTM.MatrixVectorMultiplication(wMatrix:WiT, xArray:dhsi)
            
            //dWff y dWfxh
            var dWihh = skitagLSTM.getdW(row:hs_prev, col:dhsi)
            dWihh.insert(dhsi, at: 0)
            let dWixh = skitagLSTM.getdW(row:xl_this, col:dhsi)
            let dWi = dWihh + dWixh
            let dWiT = skitagLSTM.getTranspose(Wmatrix: dWi)
            self.lstm_thetas.WiStack = skitagLSTM.matrixAccumulator(wMatrix1: self.lstm_thetas.WiStack ?? [], wMatrix2: dWiT)
            
            
            //get derivatives, dWo
            var WoT = skitagLSTM.getTranspose(Wmatrix:self.lstm_weights.WoStack ?? []) //transpose Why
            WoT.removeFirst() //remove bias
            let dXo = skitagLSTM.MatrixVectorMultiplication(wMatrix:WoT, xArray:dhso)
            
            
            //dWff y dWfxh
            var dWohh = skitagLSTM.getdW(row:hs_prev, col:dhso)
            dWohh.insert(dhso, at: 0)
            let dWoxh = skitagLSTM.getdW(row:xl_this, col:dhso)
            let dWo = dWohh + dWoxh
            let dWoT = skitagLSTM.getTranspose(Wmatrix: dWo)
            self.lstm_thetas.WoStack = skitagLSTM.matrixAccumulator(wMatrix1: self.lstm_thetas.WoStack ?? [], wMatrix2: dWoT)
            
            
            //get derivatives, dWg
            var WgT = skitagLSTM.getTranspose(Wmatrix:self.lstm_weights.WgStack ?? []) //transpose Why
            WgT.removeFirst() //remove bias
            let dXg = skitagLSTM.MatrixVectorMultiplication(wMatrix:WgT, xArray:dhsg)
            
            //dWff y dWfxh
            var dWghh = skitagLSTM.getdW(row:hs_prev, col:dhsg)
            dWghh.insert(dhsg, at: 0)
            let dWgxh = skitagLSTM.getdW(row:xl_this, col:dhsg)
            let dWg = dWghh + dWgxh
            let dWgT = skitagLSTM.getTranspose(Wmatrix: dWg)
            self.lstm_thetas.WgStack = skitagLSTM.matrixAccumulator(wMatrix1: self.lstm_thetas.WgStack ?? [], wMatrix2: dWgT)
            
            //dhs_next
            var dX = Array(repeating: 0.0, count: dXf.count)
            for h in 0...dXf.count-1{
                dX[h] = dXf[h] + dXi[h] + dXo[h] + dXg[h]
            }
            dhs_next = Array(dX[...(hsdim-1)])
            
            //dc_next
            dc_next = skitagLSTM.elementWiseMultiplication(xArray1:hsf, xArray2:dc)
        }
    }
    func lstm_update_weights(){
        
            
        let gWi = skitagLSTMback.getderivativeBackProp(W: self.lstm_weights.WiStack ?? [],
                                                       dW: self.lstm_thetas.WiStack ?? [],
                                                       event_counter: Double(self.seq_event_count),
                                                       lstm_params: self.lstm_params) //dot accumulate derivatives; one iteration
        let veli = skitagLSTMback.get_mu_velocity(gW:gWi, lstm_params: self.lstm_params)
        self.lstm_weights.WiStack = skitagLSTMback.get_thetas_updated(W:self.lstm_weights.WiStack ?? [], v:veli)
        
        
        
        let gWf = skitagLSTMback.getderivativeBackProp(W:self.lstm_weights.WfStack ?? [],
                                                       dW: self.lstm_thetas.WfStack ?? [], event_counter: Double(self.seq_event_count),
                                        lstm_params: self.lstm_params) //dot accumulate derivatives; one iteration
        let velf = skitagLSTMback.get_mu_velocity(gW:gWf, lstm_params: self.lstm_params)
        self.lstm_weights.WfStack = skitagLSTMback.get_thetas_updated(W:self.lstm_weights.WfStack ?? [], v:velf)
        
        
        
        let gWo = skitagLSTMback.getderivativeBackProp(W:self.lstm_weights.WoStack ?? [],
                                                       dW: self.lstm_thetas.WoStack ?? [], event_counter: Double(self.seq_event_count),
                                        lstm_params: self.lstm_params) //dot accumulate derivatives; one iteration
        let velo = skitagLSTMback.get_mu_velocity(gW:gWo, lstm_params: self.lstm_params)
        self.lstm_weights.WoStack = skitagLSTMback.get_thetas_updated(W:self.lstm_weights.WoStack ?? [], v:velo)
        
        
        
        let gWg = skitagLSTMback.getderivativeBackProp(W:self.lstm_weights.WgStack ?? [],
                                                       dW: self.lstm_thetas.WgStack ?? [], event_counter: Double(self.seq_event_count),
                                        lstm_params: self.lstm_params) //dot accumulate derivatives; one iteration
        let velg = skitagLSTMback.get_mu_velocity(gW:gWg, lstm_params: self.lstm_params)
        self.lstm_weights.WgStack = skitagLSTMback.get_thetas_updated(W:self.lstm_weights.WgStack ?? [], v:velg)
        
        
        
        let gWhy = skitagLSTMback.getderivativeBackProp(W:self.lstm_weights.Why ?? [],
                                                        dW: self.lstm_thetas.Why ?? [], event_counter: Double(self.seq_event_count),
                                         lstm_params: self.lstm_params) //dot accumulate derivatives; one iteration
        let velhy = skitagLSTMback.get_mu_velocity(gW:gWhy, lstm_params: self.lstm_params)
        self.lstm_weights.Why = skitagLSTMback.get_thetas_updated(W:self.lstm_weights.Why ?? [], v:velhy)
            
            // print(gWi.count, gWi[0].count, self.lstm_weights.WiStack?.count, self.lstm_weights.WiStack?[0].count)
            // print(gWf.count, gWf[0].count, self.lstm_weights.WfStack?.count, self.lstm_weights.WfStack?[0].count)
            // print(gWo.count, gWo[0].count, self.lstm_weights.WoStack?.count, self.lstm_weights.WoStack?[0].count)
            // print(gWg.count, gWg[0].count, self.lstm_weights.WgStack?.count, self.lstm_weights.WgStack?[0].count)
            // print(gWhy.count, gWhy[0].count, self.lstm_weights.Why?.count, self.lstm_weights.Why?[0].count)
    }
    
}

struct dataChart: Identifiable {
    let id = UUID()
    
    var epoch:Double
    var costv:Double
}

struct SkitagActivity: View {
    var body: some View{
        ProgressView()
            .scaleEffect(anchor: .center)
            .progressViewStyle(CircularProgressViewStyle(tint:.black))
    }
}

