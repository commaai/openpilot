dragonpilot 0.8.5-3
========================
* Added Jetson support toggle.
* Added Steering Ratio controller.
* Reduce Accel Profile to 3 modes only. (1.8s / 1.5s / 1.2s)
* Bug fixes.

dragonpilot 0.8.5-2
========================
* Added black panda simulation toggle.
* Added No GPS toggle.
* Added No Battery Toggle.
* Bug fixes.

dragonpilot 0.8.5-1
========================
* Based on openpilot 0.8.5 devel.
* 基於 openpilot 0.8.5 devel.
* Support 1+3t / C2 / Jetson Xavier NX. 
* 支持 1+3t / C2 / Jetson Xavier NX.
* No White/Grey Panda Support.
* 不支持白灰熊.

dragonpilot 0.8.4-3
========================
* 簡化 1+3t 安裝方法. (請查閱 HOWTO-ONEPLUS.md)
* Simplied 1+3t installation. (See HOWTO-ONEPLUS.md)
* 加回舊 ssh 登錄.
* Good old ssh key.
* 修復本田錯誤. (感謝 @loveloveses)
* Fixed Honda bug. (Thanks to @loveloveses)

dragonpilot 0.8.4-2
========================
* 加回可調整加速/跟車設定.
* Added back Accel/Following Profile.
* 支持 Headless Jetson Xavier NX (https://github.com/efinilan/xnxpilot.git)
* Support Headless Jetson Xavier NX (https://github.com/efinilan/xnxpilot.git)
* 支持 1+3t (需額外安裝手續)
* Support 1+3t (Require additional install procedure)
* 支持白/灰熊
* Support White/Grey Panda.

dragonpilot 0.8.4-1
========================
* 基於 openpilot 0.8.4 devel.
* Based on openpilot 0.8.4 devel.

dragonpilot 0.8.1
========================
* 基於最新 openpilot 0.8.1 devel.
* Based on latest openpilot 0.8.1 devel.
* 加入行車記錄按鈕。(感謝 @toyboxZ 提供)
* Added REC screen button. (Thanks to @toyboxZ)

dragonpilot 0.8.0
========================
* 基於最新 openpilot 0.8.0 devel.
* Based on latest openpilot 0.8.0 devel.
* 加入 git 錯誤修正。(感謝 @toyboxZ 提供)
* Added git error fix. (Thanks to @toyboxZ)

dragonpilot 0.7.10.1
========================
* HYUNDAI_GENESIS 使用 INDI 控制器。(感謝 @donfyffe 提供)
* HYUNDAI_GENESIS uses INDI controller. (Thanks to @donfyffe)
* HYUNDAI_GENESIS 加入 Cruise 按紐 和 lkMode 支援。(感謝 @donfyffe 建議)
* HYUNDAI_GENESIS added Cruise button event and lkMode feature. (Thanks to @donfyffe)
* 支援台灣版 2018 Huyndai IONIQ + smart MDPS (dp_hkg_smart_mdps) (感謝 @andy741217 提供)
* Support 2018 Taiwan Hyundai IONIQ + smart MDPS (dp_hkg_smart_mdps) (Thanks to @andy741217)
* 使用 openpilot v0.8 的模型。(感謝 @eisenheim)
* Use openpilot v0.8 model. (Thanks to @eisenheim)
* 加入 0.8 測試版的部分優化。
* Added optimizations from pre-0.8.
* 加入 dp_honda_eps_mod 設定來使用更高的扭力 (需 eps mod)。(感謝 @Wuxl_369 提供)
* Added dp_honda_eps_mod setting to enable higher torque (eps mod required). (Thanks to @Wuxl_369)
* 修正 VW 對白/灰熊的支援 (感謝 @lirudy 提供)
* Fixed issue with white/grey panda support for VW (Thanks to @lirudy)
* GENESIS_G70 優化 (感謝 @sebastian4k 提供)
* GENESIS_G70 Optimisation (Thanks to @sebastian4k)
* HYUNDAI_GENESIS 優化 (感謝 @donfyffe 提供)
* HYUNDAI_GENESIS Optimisation (Thanks to @donfyffe)
* 加入 Dynamic gas Lite。(感謝 @toyboxZ 提供)
* Added Dynamic Gas Lite. (Thanks to @toyboxZ)
* 加入來自 afa 的 Honda inspire, accord, crv SnG 優化。(感謝 @menwenliang 提供)
* Added Honda inspire, accord, crv SnG optimisation from afa fork. (Thanks to @menwenliang) 
* 加入 dp_toyota_lowest_cruise_override_vego。(感謝 @toyboxZ 提供)
* Added dp_toyota_lowest_cruise_override_vego. (Thanks to @toyboxZ)

dragonpilot 0.7.10.0
========================
* 基於最新 openpilot 0.7.10 devel.
* Based on latest openpilot 0.7.10 devel.
* 修正 Prius 特定情況下無法操控方向盤的問題。
* Fixed unable to regain Prius steering control under certain condition.
* 更新 VW MQB 的支援。(需執行 scripts/vw.sh 腳本)
* Updated support of VW MQB. (scripts/vw.sh script required)
* 新增 2018 China Toyota CHR 指紋v2。（感謝 @xiaohongcheung 提供)
* Added 2018 China Toyota CHR FPv2. (Thanks to @xiaohongcheung)
* 加入 Headunit Reloaded Android Auto App 支援。(感謝 @Ninjaa 提供)
* Added Headunit Reloaded Android Auto App Support. (Thanks to @Ninjaa)
* 優化 nanovg。(感謝 @piggy 提供)
* Optomized nanovg. (Thanks to @piggy)
* 加入 complete_setup.sh (感謝 @深鲸希西 提供)
* Added complete_setup.sh (Thanks to @深鲸希西)
* Based on latest openpilot 0.7.10 devel.
* 修正 EON 接 PC/USB 充電器時仍會自動關機的錯誤。(感謝 @小愛 回報)
* Fixed auto shutdown issue when EON connect to PC/USB Charger. (Thanks to @LOVEChen)
* HYUNDAI_GENESIS 使用 INDI 控制器。(感謝 @donfyffe 提供)
* HYUNDAI_GENESIS uses INDI controller. (Thanks to @donfyffe)

dragonpilot 0.7.8.3
========================
* VW 加入 6 分鐘時間方向盤控制限制輔助方案。(特別感謝 @actuallylemoncurd 提供代碼)
* VW added 6 minutes timebomb assist. (dp_timebomb_assist, special thanks to @actuallylemoncurd)

dragonpilot 0.7.8.2
========================
* 修正在沒網路的情況下，開機超過五分鐘的問題。
* Fixed 5+ minutes boot time issue when there is no internet connection.
* 錯誤回傳改使用 dp 的主機。
* Used dp server for error reporting.
* 更新服務改使用 gitee 的 IP 檢查連線狀態。
* updated service uses gitee IP address instead.

dragonpilot 0.7.8.1
========================
* 加入 ko-KR 翻譯。
* Added ko-KR translation.
* 加入 Honda Jade 支援。(感謝 @李俊灝)
* Added Honda Jade support. (Thanks to @lijunhao731)
* 修正 ui.cc 內存越界的問題。(感謝 @piggy 提供)
* Fixed ui.cc memory out of bound issue. (Thanks to @piggy)
* gpxd 記錄改自動存成 zip 格式。
* gpxd now store in zip format.
* 強制關閉 panda 檢查 DOS 硬體。
* Force disabled DOS hardware check in panda.

dragonpilot 0.7.8.0
========================
* 基於最新 openpilot 0.7.8 devel.
* Based on latest openpilot 0.7.8 devel.
* 加入重置 DP 設定按鈕。(感謝 @LOVEChen 建議)
* Added "Reset DP Settings" button. (Thanks to @LOVEChen)
* 將警示訊息更改為類似於概念 UI 的設計。
* Alert messages changed to concept UI alike design.
* 當 manager 出現錯誤後，按 Exit 按鈕會執行 reset_update 腳本。
* Added ability to execute reset_update.sh when press "Exit" button once manager returned errors. 

dragonpilot 0.7.7.3
========================
* 修正方向盤監控。
* Fixed steering monitor timer param.
* 修正行駛時關閉畫面導致當機的錯誤。(感謝 @salmankhan, @stevej99, @bobbydough 回報)
* Fixed screen frozen issue when "screen off while driving" toggle is enabled. (Thanks to @salmankhan, @stevej99, @bobbydough)
* 加回 Dev Mini UI 開關。(感謝 @Ninjaa 建議)
* Re-added Dev Mini UI. (Thanks to @Ninjaa)
* 新增 (dp_reset_live_parameters_on_start) 每次發車重設 LiveParameters 值。(感謝 @eisenheim)
* Added ability (dp_reset_live_param_on_start) to reset LiveParameters on each start. (Thanks @eisenheim)
* 修正同時開啟 dp_toyota_zss 和 dp_lqr 產生的錯誤。(感謝 @bobbydough)
* Fixed error cuased by enabling both dp_toyota_zss and dp_lqr at the same time. (Thanks to @bobbydough)
* 新增 (dp_gpxd) 將 GPS 軌跡導出至 GPX 格式 (/sdcard/gpx_logs/）的功能。 （感謝 @mageymoo1）
* Added ability (dp_gpxd) to export GPS track into GPX files (/sdcard/gpx_logs/). (Thanks to @mageymoo1)
* 使用德國的車道寬度估算值。 （感謝 @arne182）
* Used lane width estimate value from Germany. (Thanks to @arne182)

dragonpilot 0.7.7.2
========================
* 加入 d_poly offset。 (感謝 @ShaneSmiskol)
* Added d_poly offset. (Thanks to @ShaneSmiskol)
* 加入 ZSS 支援。(感謝 @bobbydough, @WilliamPrius 建議, @bobbydough 測試)
* Added ZSS support. (Thanks to @bobbydough, @WilliamPrius for recommendation, @bobbydough for testing)
* 加入錯誤記錄至 /sdcard/crash_logs/ (感謝 @ShaneSmiskol 提供代碼)
* Added error logs to /sdcard/crash_logs/ (Special Thanks to @ShaneSmiskol)
* 加入 LQR 控制器開關進設定畫面。
* Added LQR Controller toggle to settings.

dragonpilot 0.7.7.1
========================
* 加入 C2 風扇靜音模式。(感謝 @dingliangxue)
* Added C2 quiet fan mode. (Thanks to @dingliangxue)
* 加入「輔助換道最低啟動速度」、「自動換道最低啟動速度」設定。
* Added "Assisted Lane Change Min Engage Speed" and "Auto Lane Change Min Engage Speed" settings.
* 加入回調校介面。(感謝 @Kent)
* Re-added Dev UI. (Thanks to @Kent)
* 加入 "dp_lqr" 設定來強制使用 RAV4 的 lqr 調校。(感謝 @eisenheim)
* Added "dp_lqr" setting to force enable lqr tuning from RAV4. (Thanks to eisenheim) 

dragonpilot 0.7.7.0
========================
* 基於最新 openpilot 0.7.7 devel.
* Based on latest openpilot 0.7.7 devel.
* 當 Manager 出現錯誤時，顯示 IP 位置。(感謝 @dingliangxue)
* When Manager failed, display IP address. (Thanks to  @dingliangxue)
* 加回 sr learner 開關。
* Re-added sr learner toggle.
* 加回 加速模式 開關。
* Re-added Accel Profile toggle.
* Toyota 加入改寫最低巡航速度功能。(感謝 @Mojo)
* Added Toyota to override lowerest cruise speed. (Thanks to @Mojo)
* 介面加入盲點偵測顯示。(感謝 @wabes)
* Added BSM indicator to UI. (Thanks to @wabes)
* 加回彎道減速功能。(感謝 @Mojo)
* re-added Slow On Curve functionality. (Thanks to @Mojo)

dragonpilot 0.7.6.2
========================
* 修正無法正確關閉駕駛監控的問題。
* Fixed unable to properly turn off driver monitor issue.

dragonpilot 0.7.6.1
========================
* 基於最新 openpilot 0.7.6.1 devel.
* Based on latest openpilot 0.7.6.1 devel.
* 優化並整合 dp 服務。 (所有的設定檔已改名，請重新設定所有的功能)
* Optimized and integrated several dp services. (Settings have been renamed, please re-config all settings)
* 完全關閉 steer ratio learner。
* Completely disabled steer ratio learner.
* 移除「加速模式」。
* Removed Accel Profile.
* 加入本田皓影混電版指紋v1。(感謝 @劉駿)
* Added Honda Breeze Hybrid FPv1. (Thanks to @劉駿)
* 加入台灣版 Toyota Prius 4.5 指紋v1。(感謝 @jeekid)
* Added Taiwan Toyota Prius 4.5 FPv1. (Thanks to @jeekid)

dragonpilot 0.7.5.4
========================
* Dynamic Follow 更新模型。(感謝 @ShaneSmiskol 提供代碼、 @cgw1968 測試)
* Updated Dynamic Follow model. (Special Thanks to @ShaneSmiskol for the feature and @cgw1968 for testing)

dragonpilot 0.7.5.3
========================
* Dynamic Follow 更新至 ShaneSmiskol:stock_additions 0.7.5 版。(感謝 @ShaneSmiskol 提供代碼、 @Wei 測試)
* Updated Dynamic Follow to ShaneSmiskol:stock_additions 0.7.5. (Special Thanks to @ShaneSmiskol for the feature and @Wei for testing)
* 優化 Lexus GSH 轉向。(感謝 @簡銘佑 測試)
* Optimize Lexus GSH steering. (Thanks to @簡銘佑)
* C2 支援自動關機「DragonAutoShutdownAt」參數。(感謝 @cgw1968 建議)
* C2 to support auto shutdown "DragonAutoShutDownAt" param. (Thanks to @cgw1968)
* 修正出現「pedalPressed」的錯誤。(感謝 @Wei 回報)
* Fixed issue showing "pedalPressed" error. (Thanks to @Wei) 
* 將剎車狀熊顯示於 dp 資訊欄。
* Added brake indicator to dp infobar.
* 修正「溫度監控」燈示。
* Fixed "Temp monitor" indicator.
* 加入「方向燈取消控制」延遲控制設。(感謝 @wabes 建議)
* Added delay config to "Disable Lat Control on Blinker". (Thanks to @wabes)
* 加入巴西版 2020 Corolla Hybrid 指紋v2。(感謝 @berno22 提供)
* Added Brazil 2020 Corolla Hybrid FPv2. (Thanks to @berno22) 

dragonpilot 0.7.5.2
========================
* 加入對 VW MQB/PQ 的支援。(感謝 @dingliangxue 移植)
* Added support to VW MQB/PQ platform. (Thanks to @dingliangxue)
* 修改成 3 小時後停止供電。(感謝 @Wei 建議)
* Updated to stop charging after 3 hrs. (Thanks to @Wei)
* 移除行車記錄下的「碰撞偵測」功能。
* Removed Impact Detection in Dashcam.
* 修正開啟「Noctua 風扇」模式導致的錯誤。(感謝 @阿濤 回報)
* Fixed a bug caused by enabling "Noctua Mod". (Thanks to @阿濤)
* 修正「位智模式」無法顯示警示的問題。(感謝 @axandres 回報)
* Fixed alert issue in waze mode. (Thanks to @axandres)
* 修正無法顯示更新中圖示的問題。
* Fixed unable to display "UPDATING" icon issue. 
* 加入「允許多次自動換道」功能。(感謝 @阿濤 建議)
* Added "Allow Continuous Auto Lane Change" Toggle. (Thanks to @阿濤)
* 修正開機後設定頁面有時會錯誤的問題。(感謝 @salmankhan、@Wei 回報)
* Fixed setting page crash issue. (Thanks to @salmankhan, @Wei)
* 修正熄火後一直出現更新訊息的錯誤。(感謝 @Sky Chang 回報)
* Fixed issue that keep showing update prompt. (Thanks to @Sky Chang)

dragonpilot 0.7.5.1
========================
* 修正因同時使用「社群功能」和「自定車型」造成的加減速問題。(特別感謝 @Wei、@Sky Chang、@Han9365、@鄧育林 的測試以及回報。)
* Fixed acceleration issue caused by used of both "Community Maintain Feature" and "Custom Car Model". (Special Thanks to @Wei, @Sky Chang, @Han9365, @鄧育林)
* 新增 DragonMaxSpeedLimit 設定值 (mph)，當如果車速高於此值 op 將會停止操控。(感謝 @Anthony 建議)
* Added DragonMaxSpeedLimit parameter (mph), op will stop controlling when car speed is high than the value. (Thanks to @Anthony)
* 更新 appd 使用 cnpmjs 來下載 APKs。
* Updated appd to use cnpmjs to download APKs.
* 修正更新服務。(感謝 @Wei)
* Fixed Update Service. (Thanks to @Wei)
* 新增加拿大版 2018 Toyota Sienna LTD 指紋(v2)。(感謝 明峰 提供)
* Added Canada 2018 Toyota Sienna LTD fingerprint (v2). (Thanks to 明峰)
* 新增「通過移動網路上傳」開關
* Added Upload Over Mobile Network toggle.
* 新增「通過熱點上傳」開關
* Added Upload Over Hotspot toggle.
* 新增加拿大版 2018 Toyota Sienna LTD 指紋(v1)。(感謝 明峰 提供)
* Added Canada 2018 Toyota Sienna LTD fingerprint (v1). (Thanks to 明峰)
* 新增大陸版 Volkswagen Golf GTI 指紋 (v1)。(感謝 easyeiji 提供)
* Added China Volkswagen Golf GTI fingerprint (v1). (Thanks to easyeiji)

dragonpilot 0.7.5.0
========================
* 基於最新 openpilot 0.7.5 devel-staging.
* Based on latest openpilot 0.7.5 devel-staging.
* 更新 dp 圖示 (特別感謝 @wabes 的設計與提供)。
* Updated dp logo, special thanks to @wabes for the design.
* 簡/繁中文版和 i18n 整合成為單一版本。  
* Merged zhs/zht/i18n versions into one.
* 新增大陸版 CAMRY HYBRID 指紋v2。(感謝 @杜子腾)
* Added China Camery Hybrid FPv2. (Thanks to @杜子腾) 
* 新增台灣版 Altis HYBRID 指紋v1。(感謝 @Fish)
* Added Taiwan Altis Hybrid FPv1. (Thanks to @Fish)
* 新增行駛時關閉畫面功能。
* Added Screen off while driving feature.
* 新增倒車時關閉畫面功能。
* Added Screen off while reversing feature.
* 新增駕駛介面加入「加速模式」切換鈕。 
* Added acceleration profile toggle onto driving UI.
* 新增自定車型功能，取代指紋暫存功能。
* Replaced fingerprint cache with custom car model selector.
* 新增可調亮度。
* Added Brightness changer.
* 新增部分德語支持。(特別感謝 @arne182 提供)
* Added partial de_DE language support (Thanks to @arne182)
* 新增停車碰撞偵測記錄功能。
* Added off road impact detection to dashcam.

2020-05-06
========================
* 更新 dp 圖示 (特別感謝 @wabes 的設計與提供)。
* 中文版整合進 i18n 版。  
* 刪除指紋暫存功能。
* 新增 CAMERY HIBRID 指紋。(感謝 @杜子腾)
* 新增行駛時關閉畫面功能。
* 新增倒車時關閉畫面功能。
* 新增駕駛介面加入「加速模式」切換鈕。 
* 新增自定義車型。

2020-04-16
========================
* [DEVEL] 加入台灣版 2016 Lexus IS200t 指紋。(感謝 Philip / Cody Dai)
* [DEVEL] 加入台灣版 2016 Toyota Prius 4.5 代指紋。(感謝 Philip)
* [DEVEL] 加入台灣版 201x Toyota RAV4 4WD 指紋。(感謝 Philip)
* [DEVEL] 加入台灣版 2020 Toyota Auris w/ LTA 指紋。(感謝 Philip)
* [DEVEL] 修正 commIssue 錯誤。(感謝 Kent 協助)

2020-04-13
========================
* [DEVEL] 加入可調整 Toyota Sng 起步反應值 (DragonToyotaSngResponse)。 (特別感謝 @Wei 提供 PR)
* [DEVEL] 駕駛介面加入「動態調整車距」按鈕。(感謝 @cgw1968-5779 建議)
* [DEVEL] 更新 update script。(感謝 深鯨希西 回報)

2020-04-10
========================
* [DEVEL] 更新 panda 至最新的 comma:master 分支。
* [DEVEL] 移除所有的第三方應用改為自動下載。
* [DEVEL] 移除「啟用原廠 DSU 模式」、「安全帶檢查」、「車門檢查」開關。

2020-03-31
========================
* [DEVEL] 更新至 2020-03-31 testing 分支。

2020-03-27
========================
* [DEVEL] 更新至最新的 testing 分支:
  * 加入波蘭版 2015 Lexus NX200T 支援。(感謝 wabes 提供)
  * 調整「啟用原廠 DSU 模式」為不再需要 AHB 。(Enable Stock DSU Mode no longer requires "AHB" toggle)
  * 加入「安全帶檢查」、「車門檢查」、「檔位檢查」、「溫度檢查」開關。
  * 加入曲率學習功能 - Curvature Learner 。(感謝 zorrobyte 提供)
  * 加入大陸版 2018 Toyota Highlander 支援。(感謝 toyboxZ 提供)
  * 加入大陸版 2018 Toyota Camry 2.0 支援。(感謝 Rming 提供)
  * 加入韓文支持。(感謝 crwusiz 提供)
  * 調整 OFFROAD 主頁翻譯將 "dragonpilot" 改回 "openpilot"。

2020-03-22
========================
* [DEVEL] 更新至最新的 testing 分支。

2020-03-17
========================
* [DEVEL] 更新至最新的 testing 分支 (commaai:devel-staging 0.7.4)。
* [DEVEL] 加入動態調整車距功能。(特別感謝 @ShaneSmiskol 提供 PR)

2020-03-14
========================
* [DEVEL] 更新 pt-Br (葡萄牙語) 翻譯。(感謝 berno22 提供)
* [DEVEL] 加入自動關機開關。(感謝 Rzxd 建議)
* [DEVEL] 調高 Toyota 扭力容錯值。
* [DEVEL] 優化讀取 dp 設定值。
* [DEVEL] 加入 2019 手動 Civic 指紋。感謝 (AlexNoop 提供)
* [DEVEL] dp 功能加入對 Subaru 車系的支援。

2020-03-06
========================
* [DEVEL] 加入葡萄牙語支持。(感謝 berno22 提供)
* [DEVEL] 加入大陸 2018 Camry、2020 RAV4 指紋。(感謝 笨木匠 提供)
* [DEVEL] 建立 devel-i18n 取代 devel-en。
* [DEVEL] devel-en is deprecated, please switch to devel-i18n instead.

2020-03-04
========================
* [DEVEL] 加入顯示駕駛監控畫面。
* [DEVEL] 加入加速模式選項。(特別感謝 @arne182, @cgw1968-5779 提供 PR)
* [DEVEL] 修正 shutdownd 在 comma two 可能會不正常關機的錯誤。(感謝 @Wei, @Rzxd 回報)

2020-02-25
========================
* [DEVEL] 更新至最新的 commaai:devel (0.7.3)。

2020-02-21
========================
* [DEVEL] 更新至最新的 commaai:devel (0.7.3)。

2020-02-14
========================
* [DEVEL] 更新至最新的 commaai:devel (0.7.2)。
* [DEVEL] 修正錯誤。

2020-02-08
========================
* [DEVEL] 更新至最新的 commaai:devel (0.7.2)。
* [DEVEL] dp 功能加入對現代 (Hyundai) 車系的支援。
* [DEVEL] 加入神盾測速照相自動啟動的開關。
* [DEVEL] 更新高德地圖至 v4.5.0.600053。
* [DEVEL] 使用 0.6.6 版的更新系統。
* [DEVEL] 修正急剎問題。(感謝 kumar 提供)

2020-01-31
========================
* [DEVEL] 移除行車介面電量、溫度顯示，(修正畫面當機、黑屏問題)

2020-01-29
========================
* [DEVEL] 修正行車介面錯誤。(感謝 深鲸希西 測試；eisenheim、HeatNation 反應)

2020-01-23
========================
* [DEVEL] 加入 Steer Ratio Learner 關閉。(感謝 eisenheim 建議)
* [DEVEL] 行車介面加入電量、溫度。(感謝 eisenheim 建議)
* [DEVEL] 優化 appd。

2020-01-19
========================
* [DEVEL] 更新至最新的 commaai:devel (0.7.1)。
* [DEVEL] 調整 appd 和 ALC 邏輯。 

2020-01-14
========================
* [DEVEL] 加入開機啟動個人熱點。(感謝 eisenheim 建議)

2020-01-08
========================
* [DEVEL] 加入大陸版 2018 Lexus RX300 支援。(感謝 cafe 提供)
* [DEVEL] 加入 DragonBTG 設定。(感謝 CloudJ、低調哥、歐姓Altis車主 提供)

2019-12-31
========================
* [DEVEL-ZHS] 加回第三方應用。

2019-12-29
========================
* [DEVEL] 更新至最新的 commaai:devel (0.7.0)。
* [DEVEL] 輔助/自動變道改為可調整參數 (進階用戶)。(DragonAssistedLCMinMPH、DragonAutoLCMinMPH、DragonAutoLCDelay)
* [DEVEL-ZHS] 修正無法運行第三方應用錯誤。(感謝 深鲸希西 反應)

2019-12-18
========================
* [DEVEL] 修正自動換道邏輯。
* [DEVEL] 更新 offroad 翻譯。
* [DEVEL] 錯誤修正。
* [DEVEL] 移除美版 2017 Civic Hatchback 指紋。(與其它車型衝突)

2019-12-17
========================
* [DEVEL] 更新至最新的 commaai:devel (0.7.0)。
* [DEVEL] 加入輔助換道開關。（24mph / 40kph 以上)
* [DEVEL] 加入自動換道開關。（40mph / 65kph 以上)
* [DEVEL] 加入大陸版 2019 雷凌汽油版指紋。 (感謝 Shell 提供)
* [DEVEL] 加入大陸版 2019 卡羅拉汽油版指紋。 (感謝 Shell 提供)
* [DEVEL] 加入美版 2017 Civic Hatchback 指紋。(感謝 CFranHonda 提供)

2019-12-10
========================
* [DEVEL] 加入位智車機模式。 (Waze Mode)

2019-11-21
========================
* [DEVEL] 修正 offroad 翻譯。(感謝 鄧育林 回報)
* [DEVEL] 調整前車靜止移動偵測參數。
* [DEVEL] 前車靜止移動偵測可在未啟用 dp 時運作。

2019-11-18
========================
* [DEVEL] 修正 offroad 翻譯。(感謝 Cody、鄧育林 回報)

2019-11-18
========================
* [DEVEL] 修正 frame 翻譯。

2019-11-15
========================
* [DEVEL] 修正不會充電的錯誤。 (感謝 袁昊 反應)

2019-11-15
========================
* [DEVEL] 修正充電控制。 (感謝 KT 反應)
* [DEVEL] 更新 frame 翻譯，改為多語言版。 (感謝 深鲸希西、shaoching885、鄧育林 反應)
* [DEVEL] 更新至最新的 commaai:devel (0.6.6)。

2019-11-12
========================
* [DEVEL] 只顯示電量文字 (注意：有時不會更新，需要拔插 USB 線)
* [DEVEL] 自動偵測並鎖定硬體 (EON / UNO)。

2019-11-12
========================
* [DEVEL] 加入鎖定硬體 (EON / UNO) 的程式碼。

2019-11-11
========================
* [DEVEL] 更新高德地圖至 v4.3.0.600310 R2098NSLAE
* [DEVEL] 更新 MiXplorer 至 v6.40.3
* [DEVEL] 更新至最新的 commaai:devel (0.6.6)。
* [DEVEL] 前車靜止移動偵測加入偵測警示。

2019-11-07
========================
* [DEVEL] 讓 Bosch 系統顯示三角。 (感謝 ching885 回報)
* [DEVEL] 更新 offroad 多語言版簡體中文翻譯 (感謝 Rming 提供)

2019-11-06
========================
* [DEVEL] 修正 0.6.6 appd 和 dashcamd 錯誤。 (感謝 鄧育林 回報)
* [DEVEL] 更新至最新的 commaai:devel (0.6.6)。

2019-11-05
========================
* [DEVEL] 加入台灣 Lexus 2017 GS450h 支援。 (感謝 簡銘佑 提供指紋)

2019-11-01
========================
* [DEVEL] 新增神盾測速照相。 (感謝 Sky Chang 和 Wei Yi Chen)
* [DEVEL] 修正 offroad 翻譯。 (感謝 Leo Hsieh)

2019-11-01
========================
* [DEVEL] 移除 Miui 字型，縮小 dp 使用空間。
* [DEVEL] 更新 offroad 為多語言版
* [DEVEL] 更新至最新的 commaai:devel (0.6.5)。

2019-10-29
========================
* [DEVEL] 加入 SnG 補丁。（感謝 楊雅智)

2019-10-28
========================
* [DEVEL] 更新至最新的 commaai:devel (0.6.5)。
* [DEVEL] 調整 dragon_allow_gas 邏輯 (請回報任何問題，需更新 Panda 韌體)

2019-10-18
========================
* [DEVEL] 加入前車靜止移動偵測。(測試版，感謝 ucolchen)
* [DEVEL] 移除強迫網路連線提示。(感謝 Shell)
* [DEVEL] 修正 allow_gas 功能。

2019-10-18
========================
* [DEVEL] 加入彎道減速功能開關。
* [DEVEL] 強迫使用 dp 版 Panda 韌體。
* [DEVEL] 更新至最新的 commaai:devel (0.6.5)。 

2019-10-17
========================
* [DEVEL] 加入「車型」顯示於 dp 設定畫面。
* [DEVEL] 修正充電控制讀取預設值的錯誤。
* [DEVEL] 修正無法顯示更新記錄的錯誤。

2019-10-16
========================
* [DEVEL] 刷新 Panda 韌體按鈕將會自動重啟 EON。(感謝 鄧育林 建議)
* [DEVEL] 下載更新記錄時使用 "no-cache" 標頭。
* [DEVEL] 更新高德地圖至 v4.3.0
* [DEVEL] 刪除 bs (Branch Switcher)

2019-10-14
========================
* [DEVEL] 啟用自動更新功能。(感謝 鄧育林 提供)
* [DEVEL] 清除不再使用的 dp params。
* [DEVEL] 加入數字電量指示。(感謝 鄧育林 建議)
* [DEVEL] 加入刷新 Panda 韌體按鈕。

2019-10-11
========================
* [DEVEL] 更新至最新的 commaai:devel (0.6.5)。 
* [DEVEL] 加入台灣 2019 RAV4 汽油版指紋。 (感謝 Max Duan / CloudJ 提供)

2019-10-09
========================
* [DEVEL] 加入當 LatCtrl 關閉時，畫面顯示提示訊息。

2019-10-08
========================
* [DEVEL] 加回駕駛監控開關。
* [DEVEL] 加入 bs (branch switcher) 程式。

2019-10-07
========================
* [DEVEL] 加入台灣版 2019 RAV4H 油電版指紋。(感謝 Max Duan 提供)

2019-10-05
========================
* [DEVEL] 移除 curvature learner: 轉角明顯比原廠小。
* [DEVEL] 更新至最新的 commaai:devel (0.6.4)。 

2019-09-30
========================
* [DEVEL] 更新 curvature learner 版本至 v4。
* [DEVEL] Lexus ISH 使用更精確的 EPS Steering Angle Sensor 

2019-09-27
========================
* [DEVEL] 加入 Zorrobyte 的 curvature learner (https://github.com/zorrobyte/openpilot)
* [DEVEL] 加入可開關駕駛監控的程式碼。
* [DEVEL] 取消當 steering 出現錯誤時，自動切斷方向控制 2 秒的機制。
* [DEVEL] 讓行車介面的「方向盤」/「轉彎」圖示半透明化。

2019-09-26
========================
* [DEVEL] 修正當「啟用記錄服務」關閉時，make 會有問題的錯誤。 (感謝 shaoching885 和 afa 回報)

2019-09-24
========================
* [DEVEL] 行車介面加入可開關的「前車」、「路線」、「車道」設定。
* [DEVEL] 行車介面加入可開關的「方向燈號」提示。 (感謝 CloudJ 建議，程式碼來源: https://github.com/kegman/openpilot)

2019-09-23
========================
* [DEVEL] 優化讀取 params 的次數。
* [DEVEL] 加入可開關的車道偏移警示。
* [DEVEL] 修正充電控制邏輯。
* [DEVEL] 加入台灣 Prius 4.5 指紋。 (感謝 Lin Hsin Hung 提供)

2019-09-20
========================
* [DEVEL] 加入充電控制功能。 (感謝 loveloveses 和 KT 建議)

2019-09-16
========================
* [DEVEL] 加入台灣 CT200h 指紋。 (感謝 CloudJ 提供)
* [DEVEL] 加入美版 CT200h 移植。 (感謝 thomaspich 提供)

2019-09-13
========================
* [DEVEL] 行車介面加入可開關的「速度顯示」設定。

2019-09-09
========================
* [DEVEL] 加入 GreyPanda 模式。

2019-08-28
========================
* [DEVEL] 加入可調警示音量。

2019-08-27
========================
* [DEVEL] 自動關機改為可調時長。
