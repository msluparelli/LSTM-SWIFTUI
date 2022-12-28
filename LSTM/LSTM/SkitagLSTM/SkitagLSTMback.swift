//
//  SkitagLSTMback.swift
//  LSTM
//
//  Created by Miguel Santos Luparelli Mathieu on 6/11/22.
//

import SwiftUI

class skitagLSTMback{
    
    class func getderivativeBackProp(W:[[Double]],
                                     dW:[[Double]],
                                     event_counter:Double,
                                     lstm_params:lstmPARAMS) -> [[Double]]{
        
        let mgd = 1.0 / event_counter
        let row = dW.count
        let col = dW[0].count
        var W_grad = Array(repeating: Array(repeating: 0.0, count: col), count: row)
        
        //bias update
        for i in 0...row - 1{
            W_grad[i][0] = dW[i][0] * mgd
        }
        for i in 0...row - 1{
            for j in 1...col - 1{
                W_grad[i][j] = (mgd * dW[i][j]) + (lstm_params.lmbda * W[i][j])
            }
        }
        return W_grad
    }
    class func get_thetas_updated(W:[[Double]], v:[[Double]]) -> [[Double]] {
        
        let row = W.count
        let col = W[0].count
        var W_updated = W
        for i in 0...row - 1{
            for j in 0...col - 1{
                W_updated[i][j] += v[i][j]
            }
        }
        return W_updated
    }
    class func get_mu_velocity(gW:[[Double]], lstm_params:lstmPARAMS) -> [[Double]]{
        
        let row = gW.count
        let col = gW[0].count
        let gW_c = clipping_gradients(gW:gW)
        var velocity = Array(repeating: Array(repeating: 0.0, count: col), count: row)
        //let learning_A = ADAGRAD_learning(gW:gW)
        
        for i in 0...row - 1{
            for j in 0...col - 1{
                velocity[i][j] = (lstm_params.mu * velocity[i][j]) - (lstm_params.learning_rate * gW_c[i][j])
            }
        }
        return velocity
    }
    class func clipping_gradients(gW:[[Double]]) -> [[Double]]{
        
        let v:Double = 1.0
        let row = gW.count
        let col = gW[0].count
        var gW_C = Array(repeating: Array(repeating: 0.0, count: col), count: row)
        var sumcol = Array(repeating: 0.0, count: col)
        for i in 0...row - 1{
            for j in 0...col - 1{
                sumcol[j] += gW[i][j]
            }
        }
        let gnorm:Double = sumcol.max() ?? 1
        for i in 0...row - 1{
            for j in 0...col - 1{
                if gnorm > v{
                    gW_C[i][j] += (gW[i][j] * v) / gnorm
                } else {
                    gW_C[i][j] += gW[i][j]
                }
            }
        }
        return gW_C
    }
}

