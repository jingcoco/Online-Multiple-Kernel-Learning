

#pragma once
#include "LossFunction.h"

namespace SOL {
	template <typename FeatType, typename LabelType>
	class LogisticLoss: public LossFunction<FeatType, LabelType> {
		public:
			virtual float GetLoss(LabelType label, float predict) {
				float tmp = -predict * label;
				if (tmp > 100.f) return tmp; 
				else if (tmp < -100.f) return 0.f;
				else
					return std::log(1.f + std::exp(tmp));
			}

            //aggressive learning 
			virtual float GetGradient(LabelType label, float predict) {
				float tmp = predict * label;
				if (tmp > 100.f) //to reject numeric problems
					return 0.f;
				else if (tmp  < -100.f)
					return (float)(-label); 
				else
					return -label / (1.f + std::exp(tmp)); 
			}
 	};
}
