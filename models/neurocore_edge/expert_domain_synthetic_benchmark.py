#!/usr/bin/env python3
"""Twenty-domain synthetic corpus benchmark: native NeuroCore vs micro-agent managed.

All examples are authored, fictional templates generated locally. They are split
by example ID into train/validation sets. No web data, user data, HF datasets,
external checkpoints or foundation-model weights are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from modeling_neurocore_edge import NeuroCoreEdgeConfig, NeuroCoreEdgeForCausalLM, count_parameters
from training_micro_agents import MicroAgentTrainingController

SEED = 23
SEQ_LEN = 48
BATCH_SIZE = 2
LEARNING_RATE = 0.003
VOCAB_SIZE = 257  # byte values 0..255 shifted by one; zero is reserved.


# Three training records and two held-out records per domain. Scenarios and
# entities are invented examples, never excerpts from external datasets.
DOMAIN_CORPORA: dict[int, dict[str, Any]] = {
0: {"name":"doc_editing","train":[
("वाक्य सुधारें: मीना कल बाजार जाता है।","मीना कल बाजार जाती है।"),
("वाक्य सुधारें: हमने नई पुस्तक पढ़ता हूँ।","हमने नई पुस्तक पढ़ी।"),
("वाक्य सुधारें: बच्चे मैदान में खेल रहा हैं।","बच्चे मैदान में खेल रहे हैं।")],
"valid":[("वाक्य सुधारें: सीमा और राधा गीत गाता हैं।","सीमा और राधा गीत गाती हैं।"),
("वाक्य सुधारें: मैं कल पत्र लिखेगा।","मैं कल पत्र लिखूँगा।")]},
1: {"name":"doc_creation","train":[
("विद्यालय बैठक का संक्षिप्त सूचना-पत्र बनाएं।","सूचना: विद्यालय बैठक सोमवार को कक्ष दो में होगी। सभी अभिभावक समय पर आएँ।"),
("कार्यशाला के लिए शीर्षक और उद्देश्य लिखें।","शीर्षक: रचनात्मक लेखन कार्यशाला। उद्देश्य: विद्यार्थियों में स्पष्ट लेखन और कल्पनाशीलता बढ़ाना।"),
("पुस्तकालय कार्यक्रम का परिचय लिखें।","कार्यक्रम: पठन सप्ताह। उद्देश्य: रोज़ पढ़ने की आदत और पुस्तक चर्चा को प्रोत्साहित करना।")],
"valid":[("विज्ञान प्रदर्शनी का सूचना-पत्र बनाएं।","सूचना: विज्ञान प्रदर्शनी शुक्रवार को सभागार में लगेगी। विद्यार्थी अपने मॉडल प्रस्तुत करें।"),
("स्वच्छता अभियान का परिचय लिखें।","अभियान: स्वच्छ परिसर। उद्देश्य: कचरा पृथक्करण और स्वच्छता की आदत विकसित करना।")]},
2: {"name":"summarization","train":[
("सार लिखें: काल्पनिक नगर में वर्षा के बाद उद्यान में पौधे लगाए गए और जल-संग्रह टैंक साफ किए गए।","नगर के उद्यान में पौधारोपण हुआ और वर्षा जल-संग्रह व्यवस्था सुधारी गई।"),
("सार लिखें: एक कक्षा ने पुराने कागज़ इकट्ठे किए, उनका पुनर्चक्रण कराया और नई कॉपियाँ बनाईं।","कक्षा ने कागज़ पुनर्चक्रित करके नई कॉपियाँ तैयार कीं।"),
("सार लिखें: मोहल्ले के लोगों ने पुस्तकें साझा करने की अलमारी बनाई और हर सप्ताह सूची अपडेट की।","मोहल्ले ने पुस्तकों के आदान-प्रदान के लिए साझा अलमारी बनाई।")],
"valid":[("सार लिखें: विद्यार्थियों ने छत पर सब्ज़ियाँ उगाईं और खाद बनाने के लिए रसोई कचरे का उपयोग किया।","विद्यार्थियों ने छत पर सब्ज़ियाँ उगाईं और जैविक खाद बनाई।"),
("सार लिखें: खेल समूह ने अभ्यास का समय बदला ताकि सभी सदस्य स्कूल के बाद शामिल हो सकें।","खेल समूह ने सभी सदस्यों की सुविधा के लिए अभ्यास का समय बदला।")]},
3: {"name":"reading_qa","train":[
("पाठ: काल्पनिक द्वीप पर नीला दीपक हर शाम जलता है। प्रश्न: दीपक कब जलता है?","उत्तर: दीपक हर शाम जलता है।"),
("पाठ: रिया ने तीन बीज मिट्टी में लगाए। प्रश्न: रिया ने कितने बीज लगाए?","उत्तर: रिया ने तीन बीज लगाए।"),
("पाठ: पुस्तकालय मंगलवार को बंद रहता है। प्रश्न: पुस्तकालय किस दिन बंद रहता है?","उत्तर: पुस्तकालय मंगलवार को बंद रहता है।")],
"valid":[("पाठ: अमन ने पीली पतंग खरीदी। प्रश्न: पतंग का रंग क्या था?","उत्तर: पतंग पीली थी।"),
("पाठ: बस सुबह सात बजे रवाना हुई। प्रश्न: बस कब रवाना हुई?","उत्तर: बस सुबह सात बजे रवाना हुई।")]},
4: {"name":"read_aloud","train":[
("पाठ को वाचन-अनुकूल विरामों के साथ लिखें: आज मौसम सुहावना है चलो बाहर चलें","आज मौसम सुहावना है। चलो, बाहर चलें।"),
("पाठ को स्पष्ट उच्चारण के लिए विराम दें: नमस्कार मित्रों कार्यक्रम शुरू होता है","नमस्कार, मित्रों। कार्यक्रम शुरू होता है।"),
("वाचन के लिए वाक्य व्यवस्थित करें: पहले दीप जलाएँ फिर पुस्तक खोलें","पहले दीप जलाएँ। फिर पुस्तक खोलें।")],
"valid":[("वाचन के लिए विराम लगाएँ: वाह कितना सुंदर दृश्य है","वाह! कितना सुंदर दृश्य है।"),
("वाक्य को सहज वाचन योग्य बनाएँ: कृपया बैठें हम अभी आरंभ करेंगे","कृपया बैठें। हम अभी आरंभ करेंगे।")]},
5: {"name":"user_guidance","train":[
("नए पाठक को पुस्तक उधार लेने की प्रक्रिया बताएं।","पहले सदस्यता पर्ची भरें, फिर पुस्तक चुनें और निर्गमन काउंटर पर दर्ज कराएँ।"),
("किसी नए ऐप में मजबूत पासवर्ड बनाने के सामान्य चरण बताएं।","लंबा और अलग पासवर्ड चुनें, अक्षर-अंक-चिह्न मिलाएँ और उसे दोबारा उपयोग न करें।"),
("ऑनलाइन फॉर्म भेजने से पहले जाँच सूची दें।","नाम और विवरण जाँचें, आवश्यक फ़ील्ड भरें, संलग्नक देखें और भेजने से पहले पूर्वावलोकन करें।")],
"valid":[("कार्यक्रम में पंजीकरण करने के सामान्य चरण बताएं।","निर्देश पढ़ें, पंजीकरण विवरण भरें, जानकारी जाँचें और पुष्टि सुरक्षित रखें।"),
("डिजिटल दस्तावेज़ व्यवस्थित रखने के सुझाव दें।","स्पष्ट फ़ोल्डर बनाएँ, एकसमान फ़ाइल नाम रखें और नियमित बैकअप लें।")]},
6: {"name":"legal_contracts","train":[
("काल्पनिक किराया समझौते के लिए तटस्थ मसौदा खंड लिखें।","मसौदा: पक्षकार परिसर, अवधि, भुगतान और रखरखाव की शर्तें लिखित रूप में तय करेंगे। यह केवल उदाहरण है, कानूनी सलाह नहीं।"),
("काल्पनिक सेवा अनुबंध में कार्य-सीमा दर्ज करें।","मसौदा: सेवा प्रदाता सहमत कार्य-सूची और समयसीमा के अनुसार सेवाएँ देगा। परिवर्तन लिखित सहमति से होंगे।"),
("उदाहरण अनुबंध में गोपनीयता खंड का सार दें।","मसौदा: साझा की गई गोपनीय जानकारी का उपयोग केवल सहमत उद्देश्य के लिए होगा।")],
"valid":[("काल्पनिक आपूर्ति अनुबंध में निरीक्षण शर्त लिखें।","मसौदा: प्राप्तकर्ता निर्धारित अवधि में सामग्री की जाँच कर त्रुटियाँ लिखित रूप में बताएगा।"),
("उदाहरण में भुगतान-अवधि स्पष्ट करें।","मसौदा: भुगतान की राशि, देय तिथि और भुगतान विधि अनुबंध में स्पष्ट दर्ज होंगी।")]},
7: {"name":"legal_compliance","train":[
("काल्पनिक कार्यालय के लिए अनुपालन जाँच सूची दें।","जाँचें: नीति संस्करण, जिम्मेदार अधिकारी, प्रशिक्षण अभिलेख, शिकायत प्रक्रिया और समीक्षा तिथि।"),
("काल्पनिक डेटा नीति की समीक्षा के प्रश्न दें।","पूछें: कौन-सा डेटा लिया जाता है, उद्देश्य क्या है, पहुँच किसे है और हटाने की अवधि क्या है?"),
("अनुपालन कमी के लिए सुधार योजना बनाएं।","कमी दर्ज करें, जोखिम आँकें, जिम्मेदार व्यक्ति तय करें और समयबद्ध सुधार का प्रमाण रखें।")],
"valid":[("काल्पनिक संस्था के रिकॉर्ड ऑडिट के चरण दें।","दायरा तय करें, नमूने देखें, निष्कर्ष लिखें, सुधार सौंपें और पुनःजाँच करें।"),
("नीति परिवर्तन के बाद क्या जाँचना चाहिए?","नई प्रति, प्रभाव, अनुमोदन, सूचना और कर्मचारियों के प्रशिक्षण की पुष्टि करें।")]},
8: {"name":"professional_comms","train":[
("बैठक का समय बदलने वाला विनम्र संदेश लिखें।","नमस्कार, बैठक का समय बदलकर दोपहर तीन बजे किया गया है। असुविधा के लिए खेद है।"),
("कार्य पूरा होने की औपचारिक सूचना लिखें।","नमस्कार, निर्धारित कार्य पूरा कर लिया गया है। कृपया संलग्न विवरण देखकर प्रतिक्रिया दें।"),
("प्राप्त दस्तावेज़ की पुष्टि वाला ईमेल लिखें।","नमस्कार, दस्तावेज़ प्राप्त हो गया है। समीक्षा के बाद आवश्यक टिप्पणी साझा की जाएगी।")],
"valid":[("देरी के लिए पेशेवर क्षमा संदेश लिखें।","नमस्कार, उत्तर में हुई देरी के लिए क्षमा चाहता हूँ। अद्यतन जानकारी शीघ्र साझा करूँगा।"),
("कार्य चर्चा के लिए समय माँगें।","नमस्कार, इस विषय पर चर्चा के लिए कृपया अपनी सुविधानुसार समय बताएँ।")]},
9: {"name":"tables_ledgers","train":[
("काल्पनिक खर्चों को तालिका-पंक्तियों में रखें: कागज़ 120, पेन 80।","वस्तु | राशि; कागज़ | 120; पेन | 80; कुल | 200"),
("तीन काल्पनिक मदों का योग करें: 25, 35, 40।","मद 1 | 25; मद 2 | 35; मद 3 | 40; कुल | 100"),
("लेजर में जमा 500 और खर्च 125 दर्ज करें।","प्रारंभिक शेष | 0; जमा | 500; खर्च | 125; अंतिम शेष | 375")],
"valid":[("तालिका में 60 और 90 का योग लिखें।","मद 1 | 60; मद 2 | 90; कुल | 150"),
("काल्पनिक स्टॉक में 10 वस्तुएँ आईं और 3 निकलीं।","प्राप्त | 10; जारी | 3; शेष | 7")]},
10: {"name":"ocr_pdf_layout","train":[
("दिए गए कृत्रिम लेआउट को क्रम में लिखें: शीर्षक, अनुच्छेद, पादलेख।","शीर्षक: नमूना रिपोर्ट। अनुच्छेद: यह काल्पनिक पाठ है। पादलेख: पृष्ठ 1।"),
("पंक्ति क्रम सुरक्षित रखें: नाम-आरव, कक्षा-पाँच, समूह-नीला।","नाम: आरव। कक्षा: पाँच। समूह: नीला।"),
("काल्पनिक फॉर्म के लेबल पहचानें: दिनांक, विषय, हस्ताक्षर।","लेबल: दिनांक; विषय; हस्ताक्षर।")],
"valid":[("दो कॉलम क्रम में लिखें: वस्तु-किताब, संख्या-चार।","वस्तु: किताब। संख्या: चार।"),
("शीर्षक और पादलेख अलग करें: सूची / समाप्त।","शीर्षक: सूची। पादलेख: समाप्त।")]},
11: {"name":"hindi_indic_docs","train":[
("काल्पनिक ग्राम सभा बैठक का सूचना-पत्र बनाएं।","सूचना: ग्राम सभा की बैठक रविवार को पंचायत भवन में होगी। सभी सदस्य आमंत्रित हैं।"),
("स्थानीय पुस्तकालय के लिए हिंदी में आवेदन लिखें।","सेवा में, पुस्तकालय प्रभारी। कृपया सदस्यता हेतु मेरा आवेदन स्वीकार करें। धन्यवाद।"),
("कार्यक्रम की हिंदी कार्यवाही का संक्षिप्त प्रारूप दें।","कार्यवाही: बैठक आरंभ हुई, प्रस्ताव पढ़ा गया, सुझाव दर्ज हुए और अगली बैठक तय हुई।")],
"valid":[("काल्पनिक जल-जागरूकता शिविर की सूचना दें।","सूचना: जल-जागरूकता शिविर बुधवार को सामुदायिक भवन में आयोजित होगा।"),
("पुस्तक लौटाने की हिंदी रसीद लिखें।","रसीद: प्राप्त पुस्तक सुरक्षित लौटाई गई। दिनांक और सदस्य विवरण अभिलेख में दर्ज हैं।")]},
12: {"name":"translation","train":[
("अंग्रेज़ी से हिंदी अनुवाद करें: The room is quiet.","कमरा शांत है।"),
("अंग्रेज़ी से हिंदी अनुवाद करें: Please open the window.","कृपया खिड़की खोलें।"),
("हिंदी से अंग्रेज़ी अनुवाद करें: हम शाम को मिलेंगे।","We will meet in the evening.")],
"valid":[("अंग्रेज़ी से हिंदी अनुवाद करें: The garden is clean.","बगीचा साफ़ है।"),
("हिंदी से अंग्रेज़ी अनुवाद करें: कृपया धीरे बोलें।","Please speak slowly.")]},
13: {"name":"citations_dola","train":[
("केवल दिए गए स्रोत आईडी से दावा जोड़ें: दावा-A; स्रोत synthetic:01।","दावा-A — संदर्भ: synthetic:01 (काल्पनिक परीक्षण स्रोत)।"),
("स्रोत अनुपस्थित हो तो क्या करें?","स्रोत उपलब्ध नहीं है; दावे को अप्रमाणित चिह्नित करें और संदर्भ न गढ़ें।"),
("दो दावों के लिए अलग स्रोत आईडी दिखाएँ।","दावा-A — synthetic:02; दावा-B — synthetic:03।")],
"valid":[("बिना स्रोत वाले कथन को कैसे चिह्नित करें?","स्रोत नहीं दिया गया; सत्यापन आवश्यक।"),
("केवल synthetic:04 उपलब्ध हो तो किसे उद्धृत करें?","केवल synthetic:04 को, और उसे काल्पनिक परीक्षण संदर्भ बताकर।")]},
14: {"name":"technical_academic","train":[
("काल्पनिक ऊर्जा परियोजना के लिए तटस्थ सार लिखें।","यह नमूना परियोजना ऊर्जा उपयोग दर्ज करती है और परिणामों की तुलना के लिए मापन योजना प्रस्तुत करती है।"),
("परीक्षण रिपोर्ट की संरचना दें।","उद्देश्य, विधि, अवलोकन, सीमाएँ और निष्कर्ष को अलग-अलग अनुभागों में रखें।"),
("काल्पनिक प्रयोग की सीमा दर्ज करें।","नमूना छोटा है; परिणाम केवल इस परीक्षण परिस्थिति तक सीमित हैं।")],
"valid":[("कृत्रिम जल-गुणवत्ता परीक्षण का सार दें।","नमूना परीक्षण मापन दर्ज करता है; निष्कर्ष से पहले उपकरण और नमूने की जाँच आवश्यक है।"),
("तकनीकी लेख के लिए सीमा-वाक्य लिखें।","यह विश्लेषण सीमित कृत्रिम उदाहरणों पर आधारित है और व्यापक निष्कर्ष का दावा नहीं करता।")]},
15: {"name":"hippocampal_d2l","train":[
("सीखने के सत्र की पुनरावृत्ति योजना बनाएं।","पहले मुख्य अवधारणा पढ़ें, फिर बिना देखे याद करें और अंत में त्रुटियों की समीक्षा करें।"),
("अंतराल पुनरावृत्ति का काल्पनिक उदाहरण दें।","विषय को आज, फिर दो दिन बाद और फिर एक सप्ताह बाद दोहराएँ।"),
("सीखने के बाद स्मरण-जाँच कैसे करें?","पुस्तक बंद करके मुख्य बिंदु लिखें और मूल सामग्री से मिलाकर कमी पहचानें।")],
"valid":[("परीक्षा तैयारी का संक्षिप्त चक्र दें।","अध्ययन करें, स्वयं प्रश्न हल करें, गलतियाँ देखें और कठिन भाग दोहराएँ।"),
("किसी विषय की समझ जाँचने का तरीका दें।","विषय को अपने शब्दों में समझाएँ और उदाहरण देकर अवधारणा की जाँच करें।")]},
16: {"name":"self_correction_mnl","train":[
("वाक्य जाँचें और सुधारें: वे कल आएगा।","सुधार: वे कल आएँगे।"),
("तर्क में विरोधाभास पहचानें: सभी फूल पौधे हैं; यह पौधा फूल नहीं; इसलिए सभी पौधे फूल हैं।","सुधार: निष्कर्ष आधार-वाक्यों से नहीं निकलता; सभी पौधे फूल नहीं होते।"),
("उत्तर की कमी पहचानें: कुल 10 में से 4 गए, शेष नहीं बताया।","सुधार: शेष 6 हैं; गणना 10 - 4 = 6।")],
"valid":[("त्रुटि सुधारें: रीना और सीमा पढ़ता है।","सुधार: रीना और सीमा पढ़ती हैं।"),
("अपूर्ण निष्कर्ष पूरा करें: 12 में से 5 घटे।","सुधार: 7 शेष हैं; गणना 12 - 5 = 7।")]},
17: {"name":"coconut_planning","train":[
("पुस्तक प्रदर्शनी की तीन-चरण योजना बनाएं।","1. स्थान तय करें। 2. पुस्तकें व्यवस्थित करें। 3. आगंतुक प्रतिक्रिया दर्ज करें।"),
("छोटे अध्ययन समूह का कार्य-विभाजन करें।","समन्वयक समय तय करे, सदस्य विषय बाँटें और संपादक अंतिम सामग्री जाँचे।"),
("कार्यक्रम के जोखिम और उपाय लिखें।","जोखिम: कम उपस्थिति। उपाय: पहले सूचना दें और वैकल्पिक समय रखें।")],
"valid":[("कक्षा पठन दिवस की क्रमबद्ध योजना बनाएं।","पहले पुस्तक चुनें, फिर मौन पठन करें और अंत में चर्चा करें।"),
("छोटी बैठक की तैयारी सूची दें।","उद्देश्य लिखें, समय तय करें, सामग्री भेजें और निर्णय दर्ज करने वाला व्यक्ति चुनें।")]},
18: {"name":"bg_gate_early_exit","train":[
("सरल अनुरोध के लिए न्यूनतम मार्ग चुनें: अभिवादन।","मार्ग: एकल सरल उत्तर; अतिरिक्त एजेंट आवश्यक नहीं।"),
("अनिश्चित बहु-चरण अनुरोध के लिए मार्ग चुनें।","मार्ग: शोध और योजना एजेंट चुनें; सत्यापन के बाद उत्तर दें।"),
("एक ही विषय के दो विशेषज्ञ विकल्प हों तो क्या करें?","क्षमता-मिलान और बजट के आधार पर चुनें; अनावश्यक समानांतर कार्य न चलाएँ।")],
"valid":[("सीधे तथ्यात्मक रूपांतरण में कौन-सा मार्ग चुनें?","मार्ग: एक हल्का एजेंट और न्यूनतम सत्यापन।"),
("कठिन कार्य में संसाधन कैसे बाँटें?","मार्ग: सीमित विशेषज्ञ सक्रिय करें, समय-सीमा रखें और परिणामों की जाँच करें।")]},
19: {"name":"fabric_swarm_tools","train":[
("केवल प्रस्ताव के रूप में कैलेंडर प्रविष्टि की संरचना दें।","प्रस्ताव: {किया: कैलेंडर_प्रविष्टि, स्थिति: समीक्षा_आवश्यक}; कोई बाहरी कार्रवाई नहीं।"),
("काल्पनिक फ़ाइल सूची के लिए सुरक्षित JSON योजना दें।","{किया: फ़ाइल_सूची, पथ: नमूना/दस्तावेज़, स्थिति: केवल_योजना}"),
("किसी बाहरी टूल से पहले कौन-सी जाँचें हों?","अनुमति, लक्ष्य, दायरा और पुष्टि जाँचें; इस परीक्षण में कोई टूल निष्पादित न करें।")],
"valid":[("ईमेल भेजने की जगह केवल मसौदा योजना दें।","{किया: ईमेल_मसौदा, स्थिति: उपयोगकर्ता_समीक्षा_आवश्यक}"),
("काल्पनिक रिपोर्ट निर्यात के लिए योजना दें।","{किया: रिपोर्ट_निर्यात, प्रारूप: PDF, स्थिति: केवल_योजना}")]}
}


def build_corpora(manifest_path: Path) -> dict[int, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("expert manifest must be a list")
    ids = [row.get("expert_id") for row in manifest if isinstance(row, dict)]
    if len(ids) != len(manifest) or set(ids) != set(range(20)) or len(ids) != 20:
        raise ValueError("manifest must define unique expert IDs 0..19")
    result: dict[int, dict[str, Any]] = {}
    for expert in manifest:
        eid = expert["expert_id"]
        domain = DOMAIN_CORPORA[eid]
        if expert["name"] != domain["name"]:
            raise ValueError(f"domain mismatch for expert {eid}: {expert['name']}")
        if len(domain["train"]) < 2 or len(domain["valid"]) < 2:
            raise ValueError(f"expert {eid} needs at least two train and validation examples")
        result[eid] = {
            "expert_id": eid,
            "expert_name": domain["name"],
            "train_records": [
                {"record_id": f"expert-{eid}-train-{i}", "expert_id": eid, "input": a, "target": b,
                 "text": f"कार्य: {a}\nउत्तर: {b}\n"}
                for i, (a, b) in enumerate(domain["train"])
            ],
            "validation_records": [
                {"record_id": f"expert-{eid}-valid-{i}", "expert_id": eid, "input": a, "target": b,
                 "text": f"कार्य: {a}\nउत्तर: {b}\n"}
                for i, (a, b) in enumerate(domain["valid"])
            ],
        }
    return result


def encode_records(records: list[dict[str, str]]) -> torch.Tensor:
    raw = "\n".join(row["text"] for row in records).encode("utf-8")
    # Shift bytes by one so token ID zero remains reserved.
    return torch.tensor([byte + 1 for byte in raw], dtype=torch.long)


def build_model() -> NeuroCoreEdgeForCausalLM:
    cfg = NeuroCoreEdgeConfig(
        vocab_size=VOCAB_SIZE, hidden_size=16, num_hidden_layers=1,
        num_experts=20, experts_per_token=2, expert_intermediate_size=16,
        attention_heads=4, kv_heads=2, sliding_window=SEQ_LEN,
        recurrent_state_size=8, router_aux_loss_coefficient=0.01,
    )
    return NeuroCoreEdgeForCausalLM(cfg).cpu()


def sample_batch(stream: torch.Tensor, *, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = stream.numel() - SEQ_LEN - 1
    if max_start < 1:
        raise ValueError("encoded expert corpus is too short for the benchmark sequence length")
    starts = torch.randint(0, max_start + 1, (BATCH_SIZE,), generator=generator)
    rows = [stream[int(s):int(s) + SEQ_LEN + 1] for s in starts]
    batch = torch.stack(rows)
    return batch[:, :-1].contiguous(), batch[:, 1:].contiguous()


def _loss(model, x, y) -> torch.Tensor:
    out = model(x)
    return F.cross_entropy(out["logits"].reshape(-1, VOCAB_SIZE), y.reshape(-1)) + 0.01 * out["router_aux_loss"]


def evaluate(model, valid_streams: dict[int, torch.Tensor]) -> tuple[float, dict[str, float]]:
    model.eval()
    per_expert: dict[str, float] = {}
    with torch.no_grad():
        for eid, stream in valid_streams.items():
            x, y = sample_batch(stream, generator=torch.Generator().manual_seed(SEED + eid + 900))
            per_expert[str(eid)] = float(_loss(model, x, y).item())
    return sum(per_expert.values()) / len(per_expert), per_expert


def _state_max_abs_difference(left, right) -> float:
    diffs = [(a.detach() - b.detach()).abs().max() for a, b in zip(left.parameters(), right.parameters())]
    return float(torch.stack(diffs).max().item())


def train_trial(
    corpora: dict[int, dict[str, Any]], *, steps: int, seed: int, managed: bool,
) -> dict[str, Any]:
    gate_elapsed = 0.0
    plan = None
    if managed:
        gate_start = time.perf_counter()
        plan = MicroAgentTrainingController().create_plan(
            requested_steps=steps, max_steps=steps,
            source_kind="synthetic_only", external_data_used=False,
            user_data_used=False, dataset_ids=(), external_weights_used=False,
            checkpoint_mode="ephemeral_synthetic_only",
        )
        gate_elapsed = time.perf_counter() - gate_start
        steps = plan["allocated_steps"]

    train_streams = {eid: encode_records(data["train_records"]) for eid, data in corpora.items()}
    valid_streams = {eid: encode_records(data["validation_records"]) for eid, data in corpora.items()}
    torch.manual_seed(seed)
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    initial_val, _ = evaluate(model, valid_streams)
    generator = torch.Generator().manual_seed(seed + 1000)
    per_expert_steps = {str(eid): 0 for eid in range(20)}
    sparse_inactive_parameter_steps = 0
    train_start = time.perf_counter()
    last_loss = 0.0
    for step in range(steps):
        eid = step % 20  # ensure every domain shard is visited in round-robin order
        x, y = sample_batch(train_streams[eid], generator=generator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, x, y)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        # Sparse top-2 routing intentionally leaves unselected experts without
        # gradients on a given batch. Validate every gradient that is present.
        present_grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not present_grads or not all(bool(torch.isfinite(g).all()) for g in present_grads):
            raise RuntimeError(f"missing-all or non-finite gradients at step {step}")
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None], max_norm=1.0
        )
        if len(present_grads) < len(list(model.parameters())):
            sparse_inactive_parameter_steps += 1
        optimizer.step()
        per_expert_steps[str(eid)] += 1
        last_loss = float(loss.item())
    training_elapsed = time.perf_counter() - train_start
    final_val, per_expert_val = evaluate(model, valid_streams)
    return {
        "mode": "micro_agent_managed" if managed else "single_agent_baseline",
        "steps": steps,
        "parameter_count": count_parameters(model),
        "initial_validation_loss": initial_val,
        "final_validation_loss": final_val,
        "validation_loss_reduction_fraction": (initial_val - final_val) / initial_val,
        "last_training_loss": last_loss,
        "training_elapsed_seconds": training_elapsed,
        "micro_agent_gate_elapsed_seconds": gate_elapsed,
        "per_expert_steps": per_expert_steps,
        "steps_with_sparse_inactive_parameters": sparse_inactive_parameter_steps,
        "per_expert_validation_loss": per_expert_val,
        "micro_agent_plan": plan,
        "_model": model,
    }


def run_benchmark(report_path: Path, manifest_path: Path, *, steps: int = 40, seed: int = SEED) -> dict[str, Any]:
    if steps < 20 or steps > 200 or steps % 20 != 0:
        raise ValueError("steps must be a multiple of 20 from 20 through 200")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    wall_start = time.perf_counter()
    try:
        corpora = build_corpora(manifest_path)
        baseline = train_trial(corpora, steps=steps, seed=seed, managed=False)
        managed = train_trial(corpora, steps=steps, seed=seed, managed=True)
        parity = _state_max_abs_difference(baseline["_model"], managed["_model"])
        if parity > 1e-7:
            raise RuntimeError(f"baseline/managed training diverged: {parity}")
        total_wall = time.perf_counter() - wall_start
        per_experts = []
        for eid, data in corpora.items():
            per_experts.append({
                "expert_id": eid,
                "expert_name": data["expert_name"],
                "train_examples": len(data["train_records"]),
                "validation_examples": len(data["validation_records"]),
                "train_utf8_bytes": encode_records(data["train_records"]).numel(),
                "validation_utf8_bytes": encode_records(data["validation_records"]).numel(),
                "steps": managed["per_expert_steps"][str(eid)],
                "validation_loss": managed["per_expert_validation_loss"][str(eid)],
            })
        speed_ratio = (
            baseline["training_elapsed_seconds"] / managed["training_elapsed_seconds"]
            if managed["training_elapsed_seconds"] > 0 else None
        )
        report = {
            "status": "EXPERT_DOMAIN_SYNTHETIC_BENCHMARK_PASS",
            "experiment": "neurocore_20_expert_domain_corpus_baseline_vs_managed_v1",
            "model": "native_neurocore_reference",
            "model_config": {
                "vocab_size": VOCAB_SIZE, "hidden_size": 16, "num_hidden_layers": 1,
                "num_experts": 20, "experts_per_token": 2,
                "expert_intermediate_size": 16, "recurrent_state_size": 8,
            },
            "parameter_count": managed["parameter_count"],
            "steps_per_trial": steps,
            "seed": seed,
            "corpus": {
                "source_kind": "authored_synthetic_fictional_templates",
                "external_data_used": False,
                "user_data_used": False,
                "external_model_weights_used": False,
                "experts": per_experts,
                "total_train_examples": sum(x["train_examples"] for x in per_experts),
                "total_validation_examples": sum(x["validation_examples"] for x in per_experts),
                "split_policy": "separate hand-authored template examples; no train/validation record overlap",
            },
            "comparison": {
                "same_seed_and_initialization": True,
                "same_data_and_round_robin_schedule": True,
                "baseline": {k:v for k,v in baseline.items() if k != "_model"},
                "micro_agent_managed": {k:v for k,v in managed.items() if k != "_model"},
                "max_abs_parameter_difference": parity,
                "training_time_ratio_baseline_over_managed": speed_ratio,
                "interpretation": "A ratio above 1 means managed training was faster in this one run. Treat timing as noisy; repeat on the same hardware before claiming efficiency gains.",
            },
            "total_wall_seconds": total_wall,
            "production_training_gate": "CLOSED",
            "dataset_approval_status": "ALL_DATASETS_PENDING_REVIEW",
            "open_source_model_integrated": False,
            "checkpoint_persisted": False,
            "note": "This is a tiny synthetic engineering benchmark, not full-size NeuroCore training, domain-quality validation, convergence evidence, or a production checkpoint.",
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("expert_manifest.json"))
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = run_benchmark(args.report_path, args.manifest, steps=args.steps, seed=args.seed)
    printable = dict(result)
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
