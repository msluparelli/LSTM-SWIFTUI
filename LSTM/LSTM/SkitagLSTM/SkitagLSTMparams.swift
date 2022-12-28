//
//  SkitagLSTMparams.swift
//  LSTM
//
//  Created by Miguel Santos Luparelli Mathieu on 6/11/22.
//

//
// LSTM data
//
struct dataLSTM: Equatable {
    var iot:String?
    var features:[String]?
    var label:[Double]?
    var label_pred:[Double]?
    var features_norms:[dataNORM]?
    var features_value_raw:[[Double]]?
    var features_value_norm:[[Double]]?
}

struct dataNORM: Equatable{
    var name:String
    var mean:Double?
    var max:Double?
    var min:Double?
}


//
// LSTM objects
//
struct LSTM {
    static var HS:[[Double]] = []
    static var C:[[Double]] = []
    static var I:[[Double]] = []
    static var F:[[Double]] = []
    static var O:[[Double]] = []
    static var G:[[Double]] = []
}
struct lstmWW{
    var arch:[String]?
    var zsSeq:[[Double]]?
    var WiStack:[[Double]]?
    var WfStack:[[Double]]?
    var WoStack:[[Double]]?
    var WgStack:[[Double]]?
    var Why:[[Double]]?
}
struct lstmPARAMS {
    var mu = 0.95
    var lmbda:Double = 1e-10
    var learning_rate = 7e-2
}
struct dataLSTMtrain_eval{
    var train_values:[[[Double]]]?
    var train_labels:[Double]?
    var eval_values:[[[Double]]]?
    var eval_labels:[Double]?
}
