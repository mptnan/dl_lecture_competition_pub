# 逐次記録
## 06/11
- hydra関係、、環境切り替えのセットアップ
- とりあえず何も変えずに1エポックだけ学習してみてomnicampusへ提出。スコア0.49147
- 2エポックでスコア0.49233

## 06/13
- dataloaderをforで回すところが時間的ボトルネック
- colabでtpuは使えたり使えなかったりする
- gpu(cuda)でやる
- cuda, num_epoch=1でnum_workersを変えて実験
	<details>
	<summary>表</summary>
	
	|device|num_workers|min / epoch|
	|-|-|-|
	|cuda|2|8.23|
	||4|8.33|
	||0|10.5|
	</details>

- gdownでdataがダウンロードできないエラー
  	<details>
  	<summary>詳細</summary>
  
	```txt
	Too many users have viewed or downloaded this file recently. Please
	try accessing the file again later. If the file you are trying to
	access is particularly large or is shared with many people, it may
	take up to 24 hours to be able to view or download the file. If you
	still can't access a file after 24 hours, contact your domain
	administrator.
	```
	</details>

- TPUランタイムだとエラーで動かない
	<details>
	<summary>詳細</summary>

	```txt
	ERROR: Unknown command line flag 'xla_latency_hiding_scheduler_rerun'
	# リスタートすると治る
	```
	```txt
	2024-06-12 21:23:52.161542: E external/local_xla/xla/stream_executor/stream_executor_internal.h:177] SetPriority unimplemented for this stream.

	...

	[libprotobuf FATAL external/com_google_protobuf/src/google/protobuf/message.cc:258] File is already registered: xla/service/cpu/backend_config.proto
	terminate called after throwing an instance of 'google::protobuf::FatalException'
	what():  File is already registered: xla/service/cpu/backend_config.proto
	https://symbolize.stripped_domain/r/?trace=7b7ea21fa9fc,7b7ea21a651f&map= 
	*** SIGABRT received by PID 6388 (TID 6388) on cpu 52 from PID 6388; stack trace: ***
	PC: @     0x7b7ea21fa9fc  (unknown)  pthread_kill
		@     0x7b7db9a214f9        928  (unknown)
		@     0x7b7ea21a6520  (unknown)  (unknown)
	https://symbolize.stripped_domain/r/?trace=7b7ea21fa9fc,7b7db9a214f8,7b7ea21a651f&map=5edeb7d86db111100e979a74159a3982:7b7da9e00000-7b7db9c40ba0 
	E0612 21:23:56.773352    6388 coredump_hook.cc:447] RAW: Remote crash data gathering hook invoked.
	E0612 21:23:56.773368    6388 client.cc:272] RAW: Coroner client retries enabled (b/136286901), will retry for up to 30 sec.
	E0612 21:23:56.773373    6388 coredump_hook.cc:542] RAW: Sending fingerprint to remote end.
	E0612 21:23:56.773395    6388 coredump_hook.cc:551] RAW: Cannot send fingerprint to Coroner: [NOT_FOUND] stat failed on crash reporting socket /var/google/services/logmanagerd/remote_coredump.socket (Is the listener running?): No such file or directory
	E0612 21:23:56.773402    6388 coredump_hook.cc:603] RAW: Dumping core locally.
	E0612 21:24:01.424483    6388 process_state.cc:808] RAW: Raising signal 6 with default behavior
	```
	</details>

## 06/20
- omegaconfのバージョンを下げればcolabでのpip installでwarningを出さなくできるが、そのためにはhydra-coreのバージョンを下げる必要があり、そうするとhydra configファイルが動かなくなったりするので、バージョンは下げずにwarningは無視する
- モデルの容量がでかいので全保存するのはやめた方がいい

## 07/06
- 改善案
  - 入力(question)を分散表現に
  - 回答用にhugging_faceのコーパスを使う
  - 学習時にそれぞれの質問に対する回答例10個から学習用の回答例を毎回ランダムにサンプル
  - VQAテストの評価時の前処理を忠実に実装 https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py
  - confidenceが最上の回答のみ学習
    - 個々の結果の評価にはconfidenceに関わらず全ての回答が使われる->confidenceでのスクリーニングはむしろ逆効果
- 入力を分散表現に
- 質問のencoderとしてLSTMを実装
- 回答用にhugginf_faceのコーパスを使う
- 1epochで結果0.44234
- 3epochで結果0.17702
  - ほとんどのpredが`"<unk>"`か`"unanswerable"`になっていた
  - 回答用にhugging_faceのコーパスのみ使ったのが原因？
    - そもそもhugging_faceのコーパスを使ってtrain内に無い回答を追加したとしても、train内に無いのだからその回答を出力するように学習させることができないので意味ないのでは？
- train.jsonのanswerも回答として追加
- 3epochで0.47883
- 10epochで0.47909
  

## 07/13
- accuracyは、回答データ群の中の予想回答と同じラベルの回答の数(nとおく)によって変化する
- $\mathrm{accuracy} = \mathrm{min} (0.3n, 1)$
- スコアを上げるには回答群の中の最頻値を学習するだけでいい
- 回答群の前処理については運営の回答から、おそらく配布時のコードのもので評価される
  - 回答群の前処理には配布時のものを使うべきである
- 必要でない回答（訓練データに含まれていない回答）はそもそも学習されることもないので、除いて、予想の次元を減らす
- 回答群の最頻値が複数ある場合にはどちらを学習するか
  - それぞれの最頻値の信頼度に応じてone-hotベクトルを振り分ける(0.5, ..., 0.5, ...)など
- そもそも信頼度の低い最頻値の場合には学習すべきなのか
  - 信頼度で絞り込んだ後の最頻値を学習した方が、訓練データでのスコアは低くても、未知の評価データでのスコアは高く出るかもしれない

|onehot_type|説明|
|-|-|
|global_mode|10個の中から最頻値|
|most_confident_mode|10個の中で最もanswer_confidenceの高いものの中で最頻値|
|confidence_weighted_average: y, m, n|answer_confidenceが"yes"ならy, "no"ならn, "maybe"ならmの重みで重ね合わせる|


|n_epoch|resnet|onehot_type|VQAAccuracy local|omnicampus|datetime(utime)|
|-|-|-|-|-|-|
|1|18|global_mode|0.4691|0.47883|0712 230829|
|10|18|global_mode|0.47326|0.47883|0712 233137|
|10|18|most_confident_mode|0.4726|0.47883|0713 183933|
|10|50|most_confident_mode|0.4723|0.47883|0713 202351|

## 07/14
- text encode部分をBERTにしてみた
- 画像のRandomRotation, GCNを入れた
- 出力を見たら全部unanswerableになっていた
  - できるだけunanswerable以外の回答を学習するようにする`global_mode_except_unanswerable`
  - 正答率は2%台に

## 07/15
- omnicampus上でスコア0.47883の人が複数固まっているのはおそらく全ての予想結果が`unanswerable`になっている

## 07/18
- 学習に用いられるデータから、modeが`"unanswerable"`であるものを減らしてみたがどれだけ減らしても学習結果が`"unanswerable"`になった
- modeが`"unanswerable"`になるデータ以外でも`"unanswerable"`への学習が強く進む
- 10個の回答例にひとつも`"unanswerable"`がないものでまず学習したあと、全データで少しだけ学習する（`"unanswerable"`を学習するため）とか？

- initial commit時のmain.pyで出力の偏りを実験
- こちらでも`"unanswerable"`が大多数を占めたが、lossが自分の今のものだとだいたい10なのに対してinitial commitでは6〜ぐらい。train accuracyはおなじくらい（0.47とか）。initial commitの方が学習がうまくいっているのか、lossの計算がおかしいのか
- lossの計算でtargetで渡すものを正解ベクトルから正解ラベルに変更したが変化なし
- idxとquestionやanswerがずれている可能性も考えたが、train時に表示してみると合っていた

- 検証の結果、モデルのfc層を初期状態に戻したらlossが6〜に戻った
- fc層を不適切に、または複雑に変更したのが原因だったようだとわかった