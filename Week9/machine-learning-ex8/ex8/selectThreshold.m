function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

%testval = [ yval, pval]
%pause;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    %epsilon = 1e-20;

    predictions  = ( pval < epsilon );
    ground_truth = ( yval == 1 );
    pred_truth   = ( predictions == 1 );

    % Find true positives
    % yval == 1 && predictions == 1
    tp = ground_truth & pred_truth;
    tp_ind = find(tp);
    num_tp = sum(tp);

    % Find false positives
    % yval == 0 && predictions == 1
    fp = not(ground_truth) & pred_truth;
    fp_ind = find(fp);
    num_fp = sum(fp);

    % Find false negatives
    % yval = 1 && predicions = 0
    fn = ground_truth & not( pred_truth );
    fn_ind = find(fn);
    num_fn = sum(fn);

    % Compute prediction metrics
    precision = num_tp / ( num_tp + num_fp );
    recall    = num_tp / ( num_tp + num_fn );
    F1        = 2.0 * precision * recall / ( precision + recall );
    
    dbg = false;

    if dbg
        testvec = [ yval, ground_truth, pval, pred_truth ];
        disp(sprintf('True positives: %d', num_tp));
        testvec(tp_ind,:)


        disp(sprintf('False positives: %d', num_fp));
        testvec(fp_ind,:)

        disp(sprintf('False negatives: %d', num_fn));
        testvec(fn_ind,:)

        disp(sprintf('precision: %f recall: %f F1: %f', ...
                    precision, recall, F1));
        pause;


    end



    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
       disp(sprintf('bestEpsilon: %f  bestF1=%f', ...
            bestEpsilon, bestF1));
    end
end

end
