from django.shortcuts import render
from django.http import JsonResponse


from .prediction import predict_next_word, unique_tokens


def home(request):
    if request.method == 'POST':
        try:
            input_text = request.POST.get('input_text')

            if input_text == '':
                input_text = "he will have to look into this thing and he"
            else:
                pass
            res = predict_next_word(
                input_text,
                5
            )
            response = [unique_tokens[idx] for idx in res]
            return JsonResponse({"res":response})
        except KeyError as e:
            e = {
                "code": "KeyError",
                "msg": f"word {e} is not trained please try other words"
            }
            return JsonResponse(e)
    return render(request, 'index.html')