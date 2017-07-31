
#pragma once
#include <cmath>
#include "../common/util.h"

namespace SOL {
	template <typename FeatType, typename LabelType>
	class LossFunction {
        inline char Sign(float x) {
            if (x > 0.f) 
                return 1;
            else
                return -1;
        }

        public:
		virtual inline bool IsCorrect(LabelType label, float predict) {
            return Sign(predict) == label ? true : false;
        }

        virtual float GetLoss(LabelType label, float predict) = 0;
        virtual float GetGradient(LabelType label, float predict) = 0;

	public:
		virtual ~LossFunction(){}
    };
}
