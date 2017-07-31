

#pragma once
#include "LossFunction.h"

namespace SOL {
	template <typename FeatType, typename LabelType>
	class SquareLoss: public LossFunction<FeatType, LabelType> {
        public:
            virtual float GetLoss(LabelType label, float predict) {
                return (predict - label) * (predict - label);
            }

            virtual float GetGradient(LabelType label, float predict) {
                return 2 * (predict - label); 
            }
    };
}
